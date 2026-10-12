import unittest

import reel_summarizer as rs


class PostKeyTest(unittest.TestCase):
    def test_instagram_ignores_tracking_params_and_host_variant(self):
        keys = {
            rs.post_key("https://www.instagram.com/reel/DeNq3yzT52d/?xtok=abc"),
            rs.post_key("https://instagram.com/reel/DeNq3yzT52d/"),
            rs.post_key("https://www.instagram.com/p/DeNq3yzT52d/?igsh=xyz&utm_source=qr"),
        }
        self.assertEqual(keys, {"instagram:DeNq3yzT52d"})

    def test_instagram_shortcodes_are_case_sensitive(self):
        self.assertNotEqual(
            rs.post_key("https://www.instagram.com/reel/AbC/"),
            rs.post_key("https://www.instagram.com/reel/abc/"),
        )

    def test_youtube_link_forms_share_one_key(self):
        keys = {
            rs.post_key("https://www.youtube.com/watch?v=dQw4w9WgXcQ"),
            rs.post_key("https://www.youtube.com/watch?feature=share&v=dQw4w9WgXcQ&t=10"),
            rs.post_key("https://youtu.be/dQw4w9WgXcQ?si=abc"),
            rs.post_key("https://www.youtube.com/shorts/dQw4w9WgXcQ"),
        }
        self.assertEqual(keys, {"youtube:dQw4w9WgXcQ"})

    def test_different_posts_get_different_keys(self):
        self.assertNotEqual(
            rs.post_key("https://www.instagram.com/reel/AAA/"),
            rs.post_key("https://www.instagram.com/reel/BBB/"),
        )

    def test_other_hosts_fall_back_to_host_and_path(self):
        self.assertEqual(
            rs.post_key("https://www.facebook.com/reel/12345?mibextid=x"),
            rs.post_key("https://m.facebook.com/reel/12345/"),
        )
        self.assertNotEqual(
            rs.post_key("https://www.facebook.com/watch/?v=111"),
            rs.post_key("https://www.facebook.com/watch/?v=222"),
        )


if __name__ == "__main__":
    unittest.main()
