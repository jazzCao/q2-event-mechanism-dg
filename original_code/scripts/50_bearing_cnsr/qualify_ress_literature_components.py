"""Synthetic CPU tests for baseline cores; NOT a paper reproduction or train run."""
from pathlib import Path
import json
import sys
import hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(HERE.parent)]
from wfdg.ress_ccdg_matched.author import author_net, contrast, flatten_dim
from wfdg.mocd_model import EncoderGTable3, ProjHead, ClfHead, DecoderD, barlow_pair, softmax_weights
from wfdg.mocd_train import match_loss_fault, match_loss_domain, vat_pair

OUT = ROOT / "results/wfdg/ress_literature_qualification"


class PaperEncoder(nn.Module):
    """Table3 G has one linear after flatten, not the historical two."""
    def __init__(self):
        super().__init__()
        historical = EncoderGTable3(2048)
        self.net = historical.net
        self.fc = nn.Sequential(*list(historical.fc.children())[:3])

    def forward(self, x):
        return self.fc(self.net(x.unsqueeze(1) if x.ndim == 2 else x))


class PaperMOCD(nn.Module):
    def __init__(self, classes=3, domains=2):
        super().__init__()
        self.gf, self.gd = PaperEncoder(), PaperEncoder()
        self.pf, self.pd = ProjHead(), ProjHead()
        self.hf, self.hd = ClfHead(classes), ClfHead(domains)
        self.dec = DecoderD(2048)


def paper_or_prose_vat(model, x, labels, branch, ascent=False):
    encoder, head = (model.gf, model.hf) if branch == "f" else (model.gd, model.hd)
    query = x.detach().requires_grad_(True)
    gradient = torch.autograd.grad(F.cross_entropy(head(encoder(query)), labels), query)[0]
    perturbed = (x.detach() + (.1 if ascent else -.1) * gradient).detach()
    return F.mse_loss(encoder(perturbed), encoder(x.detach()).detach())


def run():
    torch.set_num_threads(2)
    torch.manual_seed(1600)
    checks = []
    backward_calls = 0

    def check(name, condition):
        assert bool(condition), name
        checks.append(name)

    features = torch.randn(8, 32, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1])
    authored = contrast(features, labels, .7)
    normalized = F.normalize(features, dim=1)
    similarities = normalized @ normalized.T / .7
    diagonal = torch.eye(8, dtype=torch.bool)
    positives = labels[:, None].eq(labels[None, :]) & ~diagonal
    denominator = torch.logsumexp(similarities.masked_fill(diagonal, -torch.inf), dim=1)
    oracle = -((similarities - denominator[:, None]) * positives).sum(1).div(positives.sum(1).clamp_min(1)).mean()
    check("CCDG_author_loss_equals_independent_cosine_oracle", torch.allclose(authored, oracle, atol=1e-6))
    gradient_a = torch.autograd.grad(authored, features, retain_graph=True)[0]
    gradient_b = torch.autograd.grad(oracle, features)[0]
    backward_calls += 2
    check("CCDG_author_gradient_equals_oracle", torch.allclose(gradient_a, gradient_b, atol=1e-6))
    check("CCDG_contrast_gradient_finite_nonzero", torch.isfinite(gradient_a).all() and gradient_a.abs().sum() > 0)
    for classes in (3, 8):
        featurizer, classifier = author_net(flatten_dim(2048), num_classes=classes)
        featurizer.eval(); classifier.eval()
        x = torch.randn(8, 1, 2048)
        hidden = featurizer(x)
        logits = classifier(hidden)
        check(f"CCDG_2048_input_{classes}_class_output", logits.shape == (8, classes))
        mixed_loss = .9 * F.cross_entropy(logits, labels % classes) + .1 * contrast(hidden, labels % classes, .7)
        gradient = torch.autograd.grad(mixed_loss, tuple(featurizer.parameters()))
        backward_calls += 1
        check(f"CCDG_{classes}_class_encoder_receives_actual_loss_gradient",
              all(torch.isfinite(g).all() for g in gradient) and sum(g.abs().sum() for g in gradient) > 0)
    check("CCDG_original_5120_flatten1232", flatten_dim(5120) == 1232)
    check("CCDG_adapted_2048_flatten464", flatten_dim(2048) == 464)

    model = PaperMOCD().eval()
    x = torch.log1p(torch.randn(8, 2048).abs())
    domains = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    snapshot = {n: p.detach().clone() for n, p in model.state_dict().items()}
    gf, gd = model.gf(x), model.gd(x)
    check("MOCD_Table3_G_one_linear", sum(isinstance(m, nn.Linear) for m in model.gf.fc) == 1)
    check("MOCD_both_encoders_256", gf.shape == gd.shape == (8, 256))
    check("MOCD_projection_128", model.pf(gf).shape == (8, 128))
    reconstructed = model.dec(gf, gd)
    check("MOCD_reconstructs_input_length2048", reconstructed.shape == x.shape)
    la_legacy = vat_pair(model, x, labels, domains, .1, "f") + vat_pair(model, x, labels, domains, .1, "d")
    la_formula = paper_or_prose_vat(model, x, labels, "f") + paper_or_prose_vat(model, x, domains, "d")
    la_prose = paper_or_prose_vat(model, x, labels, "f", True) + paper_or_prose_vat(model, x, domains, "d", True)
    zero = vat_pair(model, x, labels, domains, 0, "f") + vat_pair(model, x, labels, domains, 0, "d")
    backward_calls += 8
    check("MOCD_VAT_zero_epsilon_is_zero", zero.item() == 0)
    check("MOCD_three_perturbation_interpretations_finite", all(torch.isfinite(v) for v in (la_legacy, la_formula, la_prose)))
    check("MOCD_legacy_VAT_feature_penalty_active", la_legacy.item() > 0)
    losses = dict(
        lf=F.cross_entropy(model.hf(gf), labels), ld=F.cross_entropy(model.hd(gd), domains),
        lmf=match_loss_fault(model.pf(gf), labels, domains, .2),
        lmd=match_loss_domain(model.pd(gd), labels, domains, .2),
        la=la_legacy,
        lcc=match_loss_fault(.5 * F.normalize(gf, dim=-1) + .5 * F.normalize(gd, dim=-1), labels, domains, .2),
        lrs=.1 * barlow_pair(gf) + .1 * barlow_pair(gd) + .8 * ((gf.T @ gd) / len(x)).square().sum(),
        lrc=F.mse_loss(reconstructed, x),
    )
    check("MOCD_all_eight_terms_present_finite", len(losses) == 8 and all(torch.isfinite(v) for v in losses.values()))
    weights, total = softmax_weights(list(losses.values()))
    check("MOCD_negative_softmax_weights_sum_one", torch.allclose(weights.sum(), torch.tensor(1.)))
    gradient = torch.autograd.grad(total, tuple(model.parameters()), allow_unused=True)
    backward_calls += 1
    check("MOCD_total_receives_finite_nonzero_parameter_gradients",
          all(torch.isfinite(g).all() for g in gradient if g is not None) and sum(g.abs().sum() for g in gradient if g is not None) > 0)
    check("MOCD_CPU_gradient_tests_leave_weights_and_buffers_unchanged",
          all(torch.equal(value, model.state_dict()[n]) for n, value in snapshot.items()))
    toy = torch.tensor([.4, 1.1, 2.6], requires_grad=True)
    literal = (torch.softmax(-toy, dim=0) * toy).sum()
    detached = (torch.softmax(-toy.detach(), dim=0) * toy).sum()
    literal_gradient = torch.autograd.grad(literal, toy)[0]
    detached_gradient = torch.autograd.grad(detached, toy)[0]
    backward_calls += 2
    check("MOCD_loss_weight_stop_gradient_is_a_real_adaptation", not torch.allclose(literal_gradient, detached_gradient))
    files = [HERE / "wfdg/mocd_model.py", HERE / "wfdg/mocd_train.py", HERE / "wfdg/ress_e1/baseline_mocd.py",
        ROOT / "related projects/CCDG-main/domainbed/algorithms.py",
        ROOT / "related projects/CCDG-main/domainbed/lib/misc.py",
        ROOT / "related projects/CCDG-main/domainbed/TShparams_registry.py",
        ROOT / "参考文献合集/A domain generalization fault diagnosis method based on matching optimization and causal disentanglement.pdf",
        Path(__file__).resolve()]
    report = dict(status="COMPONENT_TESTS_PASS_NOT_FULL_BASELINE_QUALIFICATION",checks=checks,n_checks=len(checks),
        n_real_signal_reads=0,n_target_reads=0,n_optimizer_updates=0,synthetic_CPU_backward_or_input_gradient_calls=backward_calls,
        vat_values=dict(legacy_ascent_sign=float(la_legacy.detach()),formula_descent_raw=float(la_formula.detach()),prose_ascent_raw=float(la_prose.detach())),
        declared_gaps=["MOCD historical G has two linears; paper Table3 has one",
            "MOCD historical StageB is 20x80; paper Table4 is 20x20",
            "MOCD paper perturbation formulas and gradient-ascent prose conflict; legacy uses plus-sign FGSM",
            "MOCD historical softmax weights detach; paper does not explicitly state stop-gradient",
            "Common logFFT2048 and grouped recording split are declared adaptations, not exact original paper tables",
            "Synthetic component tests do not verify a full future baseline train/resume/freeze loop"],
        source_sha256={str(p.relative_to(ROOT)).replace(chr(92), "/"):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "COMPONENT_TESTS.json").open("w",encoding="utf-8") as file:
        json.dump(report,file,indent=2)
    print(report["status"],len(checks),report["vat_values"],flush=True)


if __name__ == "__main__":
    run()
