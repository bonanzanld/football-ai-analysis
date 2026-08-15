import unittest

from tools.qa_local_field_atlas_players import _is_full_field_support


class LocalFieldAtlasPlayerQATests(unittest.TestCase):
    def test_full_field_rectangle_is_complete(self):
        self.assertTrue(
            _is_full_field_support(
                ((0.0, 0.0), (64.0, 0.0), (64.0, 42.5), (0.0, 42.5)),
                64.0,
                42.5,
            )
        )

    def test_goal_patch_cannot_validate_full_field(self):
        self.assertFalse(
            _is_full_field_support(
                ((24.0, 0.0), (64.0, 0.0), (64.0, 42.5), (24.0, 42.5)),
                64.0,
                42.5,
            )
        )


if __name__ == "__main__":
    unittest.main()
