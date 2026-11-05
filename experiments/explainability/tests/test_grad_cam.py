"""Small synthetic tests for shared Grad-CAM output handling."""

import sys
import unittest
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.explainability.grad_cam import compute_grad_cam
from experiments.explainability.model_adapters import (
    build_model,
    extract_pneumonia_logit,
    pneumonia_probability,
    select_target_layer,
)
from experiments.explainability.run_gradcam import choose_review_indices


class TinyBinaryCNN(nn.Module):
    def __init__(self, two_logits=False):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(4, 2 if two_logits else 1)
        self.two_logits = two_logits

    def forward(self, images):
        values = self.pool(torch.relu(self.conv(images))).flatten(1)
        result = self.head(values)
        return result if self.two_logits else result.squeeze(1)

    @property
    def gradcam_target_layer(self):
        return self.conv


class GradCamTests(unittest.TestCase):
    def test_single_logit_cam_and_probability(self):
        model = TinyBinaryCNN().eval()
        inputs = torch.rand(2, 3, 32, 32)
        cam, score = compute_grad_cam(model, inputs, select_target_layer(model, "scratch_cnn"))
        self.assertEqual(tuple(cam.shape), (2, 32, 32))
        self.assertTrue(torch.isfinite(cam).all())
        self.assertTrue(((cam >= 0) & (cam <= 1)).all())
        self.assertEqual(tuple(score.shape), (2,))
        self.assertEqual(tuple(pneumonia_probability(score).shape), (2,))

    def test_two_logits_use_pneumonia_class_column(self):
        logits = torch.tensor([[4.0, -1.0], [-2.0, 3.0]])
        self.assertTrue(torch.equal(extract_pneumonia_logit(logits), torch.tensor([-1.0, 3.0])))
        probabilities = pneumonia_probability(logits)
        self.assertLess(probabilities[0].item(), 0.5)
        self.assertGreater(probabilities[1].item(), 0.5)
        model = TinyBinaryCNN(two_logits=True).eval()
        cam, scores = compute_grad_cam(model, torch.rand(2, 3, 20, 20), model.conv)
        self.assertEqual(tuple(cam.shape), (2, 20, 20))
        self.assertEqual(tuple(scores.shape), (2,))

    def test_invalid_outputs_and_targets_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "Expected model output"):
            extract_pneumonia_logit(torch.rand(2, 3, 4))
        model = TinyBinaryCNN()
        with self.assertRaisesRegex(ValueError, "contained in the model"):
            compute_grad_cam(model, torch.rand(1, 3, 20, 20), nn.Conv2d(3, 4, 3))

    def test_review_sample_is_balanced_and_includes_both_outcomes_when_available(self):
        predictions = [
            {"index": i, "true_label": label, "predicted_label": predicted}
            for i, (label, predicted) in enumerate((
                ("NORMAL", "NORMAL"), ("NORMAL", "PNEUMONIA"), ("NORMAL", "NORMAL"),
                ("PNEUMONIA", "PNEUMONIA"), ("PNEUMONIA", "NORMAL"), ("PNEUMONIA", "PNEUMONIA"),
            ))
        ]
        selected = choose_review_indices(predictions, per_class=2, seed=4)
        for label in ("NORMAL", "PNEUMONIA"):
            group = [row for row in selected if row["true_label"] == label]
            self.assertEqual(len(group), 2)
            self.assertEqual({row["true_label"] == row["predicted_label"] for row in group}, {True, False})

    def test_torchvision_architectures_have_resolvable_final_conv_layers(self):
        for architecture in ("resnet50", "densenet121"):
            with self.subTest(architecture=architecture):
                model = build_model(architecture)
                self.assertIsInstance(select_target_layer(model, architecture), nn.Conv2d)


if __name__ == "__main__":
    unittest.main()

