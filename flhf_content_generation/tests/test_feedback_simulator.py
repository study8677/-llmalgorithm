import unittest
import sys
import os

# Adjust path to import from src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from flhf_content_generation.src.feedback.feedback_simulator import FeedbackSimulator
except ModuleNotFoundError:
    # Fallback
    from src.feedback.feedback_simulator import FeedbackSimulator

class TestFeedbackSimulator(unittest.TestCase):
    def test_feedback_simulator_score(self):
        """Test score-based feedback."""
        simulator = FeedbackSimulator(feedback_type='score')
        
        # Test with ground truth
        feedback_with_gt = simulator.get_feedback(
            generated_content_1="This is a generated text.",
            ground_truth_content="This is a reference text."
        )
        self.assertIsInstance(feedback_with_gt, float, "Feedback score should be a float.")
        self.assertTrue(0.0 <= feedback_with_gt <= 1.0, "Feedback score should be between 0.0 and 1.0.")

        # Test without ground truth (should fall back to random score)
        feedback_no_gt = simulator.get_feedback(
            generated_content_1="Another generated text."
        )
        self.assertIsInstance(feedback_no_gt, float, "Feedback score (no GT) should be a float.")
        self.assertTrue(0.0 <= feedback_no_gt <= 1.0, "Feedback score (no GT) should be between 0.0 and 1.0.")
        
        # Test score with empty strings
        feedback_empty = simulator.get_feedback(
            generated_content_1="",
            ground_truth_content=""
        )
        self.assertIsInstance(feedback_empty, float)
        self.assertEqual(feedback_empty, 1.0, "Score for identical empty strings should be 1.0")

        feedback_empty_gen = simulator.get_feedback(
                    generated_content_1="",
                    ground_truth_content="Not empty"
        )
        self.assertIsInstance(feedback_empty_gen, float)


    def test_feedback_simulator_preference(self):
        """Test preference-based feedback."""
        simulator = FeedbackSimulator(feedback_type='preference')
        content1 = "This is the first option."
        content2 = "This is the second, possibly better, option."
        
        feedback = simulator.get_feedback(
            generated_content_1=content1,
            generated_content_2=content2
        )
        self.assertIn(feedback, [content1, content2], 
                      "Preference feedback should return one of the provided contents.")

        # Test that it raises error if content2 is None for preference type
        with self.assertRaises(ValueError):
            simulator.get_feedback(generated_content_1=content1)


    def test_get_preference_feedback_method(self):
        """Test the get_preference_feedback method."""
        simulator = FeedbackSimulator(feedback_type='preference') # type doesn't strictly matter for this method
        
        content_list = ["Option A", "Option B is longer", "Option C is the longest of them all"]
        
        # Test random selection when no ground truth
        preferred_random = simulator.get_preference_feedback(content_list=content_list)
        self.assertIn(preferred_random, content_list, "Preferred content must be from the list (random).")

        # Test selection based on ground truth length similarity
        ground_truth = "A medium length reference."
        preferred_gt = simulator.get_preference_feedback(content_list=content_list, ground_truth_content=ground_truth)
        self.assertIn(preferred_gt, content_list, "Preferred content must be from the list (GT based).")
        # Based on current logic, "Option B is longer" should be chosen as closest in length to ground_truth
        # len("Option A") = 8
        # len("Option B is longer") = 19
        # len("Option C is the longest of them all") = 36
        # len("A medium length reference.") = 26
        # Closest is "Option B is longer" (diff 7) or "Option C is the longest of them all" (diff 10)
        # It should be "Option B is longer" if ground_truth is "A medium length reference."
        # Actually, "Option C is the longest of them all" is closer if "A medium length reference." is used
        # Let's adjust GT to make "Option B is longer" the clear winner
        ground_truth_for_b = "This reference is close to B." # len 29
        # len("Option A") = 8 -> diff 21
        # len("Option B is longer") = 19 -> diff 10
        # len("Option C is the longest of them all") = 36 -> diff 7
        # So C is closer to "This reference is close to B."
        
        # Let's make GT very close to B
        ground_truth_for_b_exact = "Length is like Opt B" # len = 19
        preferred_gt_b = simulator.get_preference_feedback(content_list=content_list, ground_truth_content=ground_truth_for_b_exact)
        self.assertEqual(preferred_gt_b, "Option B is longer")


        # Test with empty list
        with self.assertRaises(ValueError):
            simulator.get_preference_feedback(content_list=[])


    def test_unsupported_feedback_type(self):
        """Test unsupported feedback type raises ValueError."""
        simulator = FeedbackSimulator(feedback_type='unknown_type')
        with self.assertRaises(ValueError):
            simulator.get_feedback(generated_content_1="Test")

if __name__ == '__main__':
    unittest.main()
