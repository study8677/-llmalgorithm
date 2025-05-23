import random

class FeedbackSimulator:
    """
    Simulates human feedback for generated content.

    This class can provide different types of feedback, such as numerical scores
    or preference choices between two pieces of content. The simulation logic
    is currently basic and intended as a placeholder.

    Attributes:
        feedback_type (str): The type of feedback to simulate ('score', 'preference').
    """
    def __init__(self, feedback_type='score'):
        """
        Initializes the FeedbackSimulator.

        Args:
            feedback_type (str, optional): The type of feedback to simulate.
                Supported types: 'score', 'preference'. Defaults to 'score'.
        """
        self.feedback_type = feedback_type

    def get_feedback(self, generated_content_1, generated_content_2=None, ground_truth_content=None, desired_characteristics=None):
        """
        Simulates human feedback on one or two pieces of generated content.

        The nature of the feedback depends on the `feedback_type` attribute.
        - 'score': Returns a numerical score, potentially based on `ground_truth_content`.
        - 'preference': Returns one of `generated_content_1` or `generated_content_2`,
                        simulating a choice.

        Args:
            generated_content_1 (str): The first piece of generated content.
            generated_content_2 (str, optional): The second piece of generated content,
                                                 required if `feedback_type` is 'preference'.
                                                 Defaults to None.
            ground_truth_content (str, optional): The ground truth or reference content,
                                                  used for 'score' calculation if available.
                                                  Defaults to None.
            desired_characteristics (dict, optional): A dictionary of desired
                                                      characteristics for the content (e.g.,
                                                      {'style': 'formal', 'length': 'short'}).
                                                      Currently unused. Defaults to None.

        Returns:
            float or str: The simulated feedback.
                          - If `feedback_type` is 'score', returns a float score (0.0-1.0).
                          - If `feedback_type` is 'preference', returns the chosen string content.

        Raises:
            ValueError: If `feedback_type` is 'preference' and `generated_content_2` is None,
                        or if `feedback_type` is unsupported.
        """
        if self.feedback_type == 'score':
            # Placeholder for more sophisticated scoring logic
            # e.g., using ROUGE scores if ground_truth_content is a summary
            # e.g., checking against desired_characteristics like length, style, keywords

            if ground_truth_content:
                # Simple score based on length similarity to ground truth
                len_generated = len(generated_content_1)
                len_ground_truth = len(ground_truth_content)
                if len_ground_truth == 0 and len_generated == 0:
                    return 1.0
                if len_ground_truth == 0:
                    return 0.0 # Avoid division by zero if ground truth is empty but generated is not
                score = 1.0 - min(abs(len_generated - len_ground_truth) / len_ground_truth, 1.0)
                return score
            else:
                # Fallback to a random score if no ground truth is provided
                return random.uniform(0.0, 1.0)

        elif self.feedback_type == 'preference':
            # Placeholder for more sophisticated preference logic
            # e.g., comparing content against desired_characteristics or a reference model

            if generated_content_2 is None:
                raise ValueError("generated_content_2 must be provided for 'preference' feedback type.")

            # Simple random preference
            return random.choice([generated_content_1, generated_content_2]) # Or return 0 or 1

        else:
            raise ValueError(f"Unsupported feedback type: {self.feedback_type}")

    def get_preference_feedback(self, content_list, ground_truth_content=None, desired_characteristics=None):
        """
        Simulates human feedback by selecting the preferred content from a list.

        This method is useful when comparing multiple generated content options.
        If `ground_truth_content` is provided, preference is based on length similarity.
        Otherwise, a random choice is made.

        Args:
            content_list (list of str): A list of generated content pieces to compare.
            ground_truth_content (str, optional): The ground truth or reference content.
                                                  If provided, used to determine preference
                                                  based on length similarity. Defaults to None.
            desired_characteristics (dict, optional): A dictionary of desired
                                                      characteristics for the content.
                                                      Currently unused. Defaults to None.

        Returns:
            str: The preferred content string from the `content_list`.

        Raises:
            ValueError: If `content_list` is empty.
        """
        if not content_list:
            raise ValueError("Content list cannot be empty for preference feedback.")

        # Placeholder for more sophisticated preference selection
        # For now, randomly select one or select based on proximity to ground_truth_content length
        if ground_truth_content:
            best_content = None
            min_diff = float('inf')
            for content in content_list:
                diff = abs(len(content) - len(ground_truth_content))
                if diff < min_diff:
                    min_diff = diff
                    best_content = content
            return best_content
        else:
            return random.choice(content_list)
