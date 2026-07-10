import unittest

from reward import extract_svg, reward, reward_fn, score_svg


VALID = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">
<rect x="32" y="32" width="192" height="192" rx="24" fill="#0000ff"/>
<circle cx="128" cy="128" r="45" fill="white"/>
<text x="128" y="140" text-anchor="middle" fill="black">LY</text>
</svg>"""


class RewardTests(unittest.TestCase):
    def test_valid_svg_scores_high(self):
        result = score_svg('A blue circular logo with the text "LY"', VALID)
        self.assertTrue(result["valid_xml"])
        self.assertGreaterEqual(result["total"], 0.85)
        self.assertEqual(result["fidelity"], 1.0)

    def test_extracts_svg_from_markdown_and_prose(self):
        wrapped = f"Here it is:\n```svg\n{VALID}\n```"
        self.assertEqual(extract_svg(wrapped), VALID)
        self.assertGreater(reward("blue circle", wrapped), 0.7)

    def test_missing_svg_is_zero(self):
        result = score_svg("a logo", "I cannot draw that")
        self.assertEqual(result["total"], 0.0)
        self.assertIn("no complete SVG found", result["reasons"])

    def test_self_closing_svg_is_valid_but_has_no_drawing(self):
        result = score_svg("a logo", '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"/>')
        self.assertTrue(result["valid_xml"])
        self.assertEqual(result["visible_elements"], 0)
        self.assertGreater(result["total"], 0.0)
        self.assertLessEqual(result["total"], 0.2)

    def test_malformed_xml_is_capped(self):
        result = score_svg("a logo", "<svg><rect></svg>")
        self.assertLessEqual(result["total"], 0.1)
        self.assertFalse(result["valid_xml"])

    def test_no_visible_drawing_is_capped(self):
        svg = '<svg viewBox="0 0 256 256"><rect width="10" height="10" fill="none"/></svg>'
        result = score_svg("rectangle", svg)
        self.assertEqual(result["visible_elements"], 0)
        self.assertLessEqual(result["total"], 0.2)

    def test_default_black_fill_is_visible_but_color_component_is_partial(self):
        svg = '<svg viewBox="0 0 10 10"><circle cx="5" cy="5" r="4"/></svg>'
        result = score_svg("circle", svg)
        self.assertEqual(result["visible_elements"], 1)
        self.assertIn("no explicit visible fill/stroke color", result["reasons"])

    def test_dangerous_content_is_zero(self):
        svg = '<svg viewBox="0 0 10 10"><script>alert(1)</script><circle r="2"/></svg>'
        result = score_svg("circle", svg)
        self.assertEqual(result["total"], 0.0)
        self.assertTrue(any("dangerous tag" in reason for reason in result["reasons"]))

    def test_external_reference_is_zero_but_internal_use_is_allowed(self):
        external = '<svg viewBox="0 0 10 10"><image href="https://example.com/a.png"/></svg>'
        internal = '<svg viewBox="0 0 10 10"><defs><circle id="c" r="2"/></defs><use href="#c"/></svg>'
        self.assertEqual(reward("logo", external), 0.0)
        internal_result = score_svg("circle", internal)
        self.assertGreater(internal_result["total"], 0.2)
        self.assertEqual(internal_result["visible_elements"], 1)

    def test_shapes_inside_defs_are_not_visible_without_use(self):
        svg = '<svg viewBox="0 0 10 10"><defs><circle id="c" r="2" fill="red"/></defs></svg>'
        result = score_svg("red circle", svg)
        self.assertEqual(result["visible_elements"], 0)
        self.assertLessEqual(result["total"], 0.2)

    def test_external_css_is_zero(self):
        svg = '<svg viewBox="0 0 10 10"><style>@import url(https://example.com/x.css);</style><rect width="2" height="2"/></svg>'
        result = score_svg("rectangle", svg)
        self.assertEqual(result["total"], 0.0)
        self.assertTrue(any("CSS" in reason for reason in result["reasons"]))

    def test_invalid_and_extreme_path_are_penalized(self):
        invalid = '<svg viewBox="0 0 256 256"><path d="M 0 0 X 5 5" fill="red"/></svg>'
        extreme = '<svg viewBox="0 0 256 256"><path d="M 999999 0 L 1 1 Z" fill="red"/></svg>'
        self.assertLess(score_svg("red path", invalid)["canvas"], 1.0)
        self.assertLess(score_svg("red path", extreme)["canvas"], 1.0)

    def test_negative_size_is_penalized(self):
        svg = '<svg viewBox="0 0 10 10"><rect width="-2" height="5" fill="red"/></svg>'
        self.assertLess(score_svg("red rectangle", svg)["canvas"], 1.0)

    def test_fidelity_does_not_credit_id_title_or_class(self):
        svg = '<svg viewBox="0 0 10 10"><title>blue circle LY</title><rect id="blue-circle-LY" class="blue" width="5" height="5" fill="red"/></svg>'
        result = score_svg('blue circle with text "LY"', svg)
        self.assertEqual(result["fidelity"], 0.0)

    def test_repeated_elements_are_penalized(self):
        circles = "".join('<circle cx="1" cy="1" r="1" fill="red"/>' for _ in range(10))
        svg = f'<svg viewBox="0 0 10 10">{circles}</svg>'
        result = score_svg("red circles", svg)
        self.assertLess(result["safety"], 1.0)
        self.assertTrue(any("repetitions" in reason for reason in result["reasons"]))

    def test_batch_wrapper(self):
        scores = reward_fn(["blue circle", "logo"], [VALID, "bad"])
        self.assertEqual(len(scores), 2)
        self.assertGreater(scores[0], scores[1])
        with self.assertRaises(ValueError):
            reward_fn(["one"], [])


if __name__ == "__main__":
    unittest.main()
