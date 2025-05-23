import unittest
import sys
import os
import time

# Adjust path to import from src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from flhf_content_generation.src.llm_api_simulator import LLMAPISimulator
except ModuleNotFoundError:
    # Fallback
    from src.llm_api_simulator import LLMAPISimulator

class TestLLMAPISimulator(unittest.TestCase):
    def setUp(self):
        """Initialize the LLMAPISimulator with no latency for tests."""
        self.simulator = LLMAPISimulator(api_latency=0) 

    def test_generate_summary_prompt(self):
        """Test that a prompt containing 'summarize' gets a summary response."""
        response = self.simulator.generate("summarize this document")
        self.assertIn("simulated summary", response.lower(), 
                      "Response should indicate a summary for 'summarize' prompt.")
        
        response_caps = self.simulator.generate("SUMMARY OF THE ARTICLE")
        self.assertIn("simulated summary", response_caps.lower(),
                      "Response should indicate a summary for 'SUMMARY' prompt (case-insensitive).")

    def test_generate_generic_prompt(self):
        """Test that a generic prompt gets a generic LLM response."""
        response = self.simulator.generate("tell me a story")
        self.assertIn("simulated llm response", response.lower(),
                      "Response should be a generic one for non-specific prompts.")

    def test_generate_keyword_prompts(self):
        """Test prompts with 'generate' or 'write' keywords."""
        response_gen = self.simulator.generate("generate a list of ideas")
        self.assertIn("generated text based on your prompt", response_gen.lower(),
                      "Response for 'generate' keyword is incorrect.")
        
        response_write = self.simulator.generate("write an essay about AI")
        self.assertIn("generated text based on your prompt", response_write.lower(),
                      "Response for 'write' keyword is incorrect.")

    def test_api_latency_simulation(self):
        """Test that API latency simulation works (approximately)."""
        latency_duration = 0.05 # Short duration for testing
        simulator_with_latency = LLMAPISimulator(api_latency=latency_duration)
        
        start_time = time.time()
        simulator_with_latency.generate("any prompt")
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        
        # Check if elapsed time is close to the specified latency
        # Allow for some minor overhead and timing inaccuracies
        self.assertGreaterEqual(elapsed_time, latency_duration * 0.9, 
                                "Elapsed time should be at least close to the simulated latency.")
        # Be a bit lenient on the upper bound due to system factors
        self.assertLess(elapsed_time, latency_duration * 2.0 + 0.01, 
                        "Elapsed time seems too long for the simulated latency.")

    def test_empty_prompt(self):
        """Test how the simulator handles an empty prompt."""
        response = self.simulator.generate("")
        # The current implementation will result in a generic response
        self.assertIn("simulated llm response to: ''", response.lower(), # Current generic response includes the prompt
                      "Response for empty prompt is not the expected generic one.")


if __name__ == '__main__':
    unittest.main()
```
