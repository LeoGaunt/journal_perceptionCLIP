"""
utils.py — Model-agnostic zero-shot classifier construction.

The key change from the original: every tokenisation call goes through
``clip_model.tokenize(texts)`` instead of ``clip.tokenize(texts)``.
This means the same functions work for original CLIP, OpenCLIP,
MetaCLIP, and SigLIP without any modification.
"""

import torch
from tqdm import tqdm
import random
import string
from itertools import product

from src.models.modeling import ClassificationHead


# ---------------------------------------------------------------------------
# Classifier construction
# ---------------------------------------------------------------------------

def get_zeroshot_classifier(args, clip_model, classnames, template):
    """
    Standard zero-shot head: average embeddings across templates per class.
    clip_model must expose .tokenize() and .encode_text().
    """
    logit_scale = clip_model.logit_scale
    device = args.device
    clip_model.eval()
    clip_model.to(device)

    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [t(classname) for t in template]
            token_ids  = clip_model.tokenize(texts).to(device)
            embeddings = clip_model.encode_text(token_ids)           # already normalised
            embeddings = embeddings.mean(dim=0, keepdim=True)
            embeddings = embeddings / embeddings.norm()
            zeroshot_weights.append(embeddings)

        zeroshot_weights = torch.cat(zeroshot_weights, dim=0).to(device)
        zeroshot_weights *= logit_scale.exp()

    return ClassificationHead(normalize=True, weights=zeroshot_weights)


def get_zeroshot_classifier_flat_advance(args, clip_model, classnames, template_list):
    """
    PerceptionCLIP classifier: one embedding per (class, context-combo) pair.
    template_list is a list of lists of functions — one inner list per
    context combination, each inner list contains one or more phrasings.
    """
    logit_scale = clip_model.logit_scale
    device = args.device
    clip_model.eval()
    clip_model.to(device)

    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            for template in template_list:
                texts = [t(classname) for t in template]
                token_ids  = clip_model.tokenize(texts).to(device)
                embeddings = clip_model.encode_text(token_ids)
                embeddings = embeddings.mean(dim=0, keepdim=True)
                embeddings = embeddings / embeddings.norm()
                zeroshot_weights.append(embeddings)

        zeroshot_weights = torch.cat(zeroshot_weights, dim=0).to(device)
        zeroshot_weights *= logit_scale.exp()

    return ClassificationHead(normalize=True, weights=zeroshot_weights)


def get_zeroshot_classifier_puretext_advance(args, clip_model, template_texts_list):
    """
    Factor-only head (infer_mode=1): embeds context descriptions without class.
    """
    device = args.device
    clip_model.eval()
    clip_model.to(device)
    logit_scale = clip_model.logit_scale

    with torch.no_grad():
        zeroshot_weights = []
        for template_texts in template_texts_list:
            token_ids  = clip_model.tokenize(template_texts).to(device)
            embeddings = clip_model.encode_text(token_ids)
            embeddings = embeddings.mean(dim=0, keepdim=True)
            embeddings = embeddings / embeddings.norm()
            zeroshot_weights.append(embeddings)

        zeroshot_weights = torch.cat(zeroshot_weights, dim=0).to(device)
        zeroshot_weights *= logit_scale.exp()

    return ClassificationHead(normalize=True, weights=zeroshot_weights)


# ---------------------------------------------------------------------------
# Template composition (unchanged logic, just moved here for clarity)
# ---------------------------------------------------------------------------

def generate_composite_factors(templates, selected_factors=None):
    """
    Recursively expand a factor template dict into all (value, description) combos.

    templates = {
        "factor_1": {
            "value_a": ["desc1", "desc2"],
            "value_b": ["desc1"],
        },
        "factor_2": { ... }
    }
    Returns a flat dict: { "value_a_value_x": ["combined desc", ...], ... }
    """
    if selected_factors:
        templates = {k: templates[k] for k in selected_factors if k in templates}

    if not templates:
        return {"": [""]}

    key, sub_dict = next(iter(templates.items()))
    rest_templates = {k: v for k, v in templates.items() if k != key}

    composite = {}
    for sub_key, values in sub_dict.items():
        for rest_key, rest_values in generate_composite_factors(rest_templates).items():
            new_key = f"{sub_key}_{rest_key}" if rest_key else sub_key
            combined = [
                v + (", " + rv if rv else "") if v else rv
                for v in values
                for rv in rest_values
            ]
            composite[new_key] = combined

    return composite


def compose_template(org_templates, factor_templates):
    """
    Build a list-of-lists of lambda functions for the PerceptionCLIP head.

    Each outer list is one context combination.
    Each inner list contains one or more phrasings of that combination.
    """
    factor_list = [factor_templates[k] for k in factor_templates]
    new_templates = []
    for descriptions in factor_list:
        new_factors = []
        for desc in descriptions:
            if desc:
                new_factors.append(
                    lambda c, main=org_templates[0], d=desc: main(c) + ", " + d + "."
                )
            else:
                new_factors.append(
                    lambda c, main=org_templates[0]: main(c) + "."
                )
        new_templates.append(new_factors)
    return new_templates


# ---------------------------------------------------------------------------
# Template utilities
# ---------------------------------------------------------------------------

def template_convert(template, convert_text):
    """Materialise lambda templates to plain strings (for infer_mode=1)."""
    if callable(template[0]):
        return [f(convert_text) for f in template]
    elif isinstance(template[0], list):
        return [[f(convert_text) for f in inner] for inner in template]
    raise ValueError("Invalid template format")


def randomize_text(text):
    words = text.split()
    return " ".join(
        "".join(random.choices(string.ascii_letters + string.digits, k=len(w)))
        for w in words
    )


def randomize_function(template):
    sentence = template("")
    if "," not in sentence:
        return template
    first_part, rest = sentence.split(",", 1)
    rest_start, _ = rest.split(".", 1)
    randomized = randomize_text(rest_start)
    def new_function(c, _t=template, _r=randomized):
        original = _t(c)
        fp, _ = original.rsplit(",", 1)
        return f"{fp}, {_r}."
    return new_function


def randomize_template(template_list):
    return [
        randomize_text(t) if isinstance(t, str) else randomize_function(t)
        for t in template_list
    ]


# ---------------------------------------------------------------------------
# Group accuracy (unchanged)
# ---------------------------------------------------------------------------

def group_accuracy(args, output, target, attribute):
    with torch.no_grad():
        batch_size  = target.size(0)
        exact_acc   = output.eq(target.view_as(output)).squeeze()
        group_correct = torch.zeros((args.num_attrs, args.num_labels))
        group_cnt     = torch.zeros((args.num_attrs, args.num_labels))
        for g in range(args.num_attrs):
            for y in range(args.num_labels):
                mask = (attribute == g) * (target == y)
                group_correct[g, y] = exact_acc[mask].sum()
                group_cnt[g, y]     = mask.sum()
        if batch_size != group_cnt.sum().item():
            raise ValueError("Error in group accuracy computation.")
    return group_correct, group_cnt
