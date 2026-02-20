"""Tests for the GenAI module."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from python.genai.client import GenAIClient
from python.genai.music_generator import MusicGenerator
from python.genai.prompts import (
    PromptContext,
    build_prompt,
    build_rhythmic_prompt,
    build_texture_prompt,
    build_variation_prompt,
)


class TestPrompts(unittest.TestCase):
    def test_rhythmic_prompt_contains_bpm(self) -> None:
        ctx = PromptContext(mode="rhythmic", description="hard techno kick", bpm=140.0)
        prompt = build_rhythmic_prompt(ctx)
        self.assertIn("140", prompt)
        self.assertIn("rhythmic", prompt.lower())

    def test_texture_prompt_no_bpm(self) -> None:
        ctx = PromptContext(mode="texture", description="dark ambient pad")
        prompt = build_texture_prompt(ctx)
        self.assertIn("texture", prompt.lower())
        self.assertIn("dark ambient pad", prompt)

    def test_build_prompt_dispatches(self) -> None:
        rhythmic_ctx = PromptContext(mode="rhythmic", description="test")
        texture_ctx = PromptContext(mode="texture", description="test")
        self.assertIn("rhythmic", build_prompt(rhythmic_ctx).lower())
        self.assertIn("texture", build_prompt(texture_ctx).lower())

    def test_variation_prompt_includes_index(self) -> None:
        spec = "Some original spec content"
        prompt = build_variation_prompt(spec, 2, "rhythmic")
        self.assertIn("2", prompt)
        self.assertIn(spec, prompt)

    def test_prompt_includes_key_and_genre(self) -> None:
        ctx = PromptContext(mode="rhythmic", description="groovy beat", key="Am", genre="techno")
        prompt = build_rhythmic_prompt(ctx)
        self.assertIn("Am", prompt)
        self.assertIn("techno", prompt)


class TestGenAIClient(unittest.TestCase):
    def test_init_raises_on_empty_key(self) -> None:
        with self.assertRaises(ValueError):
            GenAIClient(api_key="")

    @patch("python.genai.client.genai")
    def test_generate_text_returns_stripped_text(self, mock_genai: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value.text = "  hello world  "
        mock_genai.Client.return_value = mock_client
        client = GenAIClient(api_key="test-key")
        result = client.generate_text("test prompt")
        self.assertEqual(result, "hello world")

    @patch("python.genai.client.genai")
    def test_model_name_property(self, mock_genai: MagicMock) -> None:
        mock_genai.Client.return_value = MagicMock()
        client = GenAIClient(api_key="test-key", model_name="gemini-pro")
        self.assertEqual(client.model_name, "gemini-pro")


class TestMusicGenerator(unittest.TestCase):
    def _make_client(self) -> MagicMock:
        client = MagicMock(spec=GenAIClient)
        client.generate_with_config.return_value = "Generated spec text"
        return client

    def test_generate_spec_calls_client(self) -> None:
        client = self._make_client()
        gen = MusicGenerator(client)
        spec = gen.generate_spec("techno loop", mode="rhythmic", bpm=130.0)
        self.assertEqual(spec, "Generated spec text")
        client.generate_with_config.assert_called_once()

    def test_generate_variations_returns_correct_count(self) -> None:
        client = self._make_client()
        gen = MusicGenerator(client)
        variations = gen.generate_variations("base spec", count=3, mode="texture")
        self.assertEqual(len(variations), 3)
        self.assertEqual(client.generate_with_config.call_count, 3)

    def test_generate_pack_includes_base_plus_variations(self) -> None:
        client = self._make_client()
        gen = MusicGenerator(client)
        specs = gen.generate_pack("ambient pad", mode="texture", variation_count=2)
        self.assertEqual(len(specs), 3)
        self.assertEqual(client.generate_with_config.call_count, 3)

    def test_generate_pack_zero_variations(self) -> None:
        client = self._make_client()
        gen = MusicGenerator(client)
        specs = gen.generate_pack("test", mode="rhythmic", variation_count=0)
        self.assertEqual(len(specs), 1)


if __name__ == "__main__":
    unittest.main()
