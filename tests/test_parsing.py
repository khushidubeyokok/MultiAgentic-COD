import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.utils import strip_thoughts, parse_best_json

def test_strip_thoughts():
    print("Testing strip_thoughts()...")
    
    # Test case 1: Closed think tag
    input1 = "<think>some long reasoning</think>\n{\"diagnosis\": \"Malaria\"}"
    expected1 = "{\"diagnosis\": \"Malaria\"}"
    result1 = strip_thoughts(input1)
    print(f"Test 1: {'PASSED' if result1 == expected1 else 'FAILED'}")
    if result1 != expected1:
        print(f"  Input: {input1}")
        print(f"  Expected: {expected1}")
        print(f"  Result: {result1}")

    # Test case 2: Unclosed think tag
    input2 = "<think>reasoning starts but never ends...\n{\"diagnosis\": \"Sepsis\"}"
    expected2 = "" # Because <think>.* matches everything after it
    # Actually, if it's unclosed, we might want to keep the JSON if it's there?
    # Spec says: "handle the case where a <think> block is opened but never closed (partial output)"
    # Usually LLMs put JSON AFTER the reasoning. If reasoning is unclosed, JSON is likely missing.
    # If the JSON is INSIDE the unclosed tag, it gets stripped.
    result2 = strip_thoughts(input2)
    print(f"Test 2 (Unclosed): Result was '{result2}'")

    # Test case 3: Other XML tags
    input3 = "<reasoning>Logic here</reasoning><analysis>More logic</analysis>{\"diagnosis\": \"Pneumonia\"}"
    expected3 = "{\"diagnosis\": \"Pneumonia\"}"
    result3 = strip_thoughts(input3)
    print(f"Test 3: {'PASSED' if result3 == expected3 else 'FAILED'}")
    if result3 != expected3:
        print(f"  Result: {result3}")

def test_parse_best_json():
    print("\nTesting parse_best_json()...")
    
    # Test case: Largest valid object with diagnosis key
    # Input has a small JSON fragment and a larger one. 
    # The larger one should be picked.
    input1 = """
    Some text here.
    { "partial": "data" }
    More text.
    {
      "diagnosis": "Malaria",
      "scores": { "Malaria": 5, "Sepsis": 1 }
    }
    """
    result1 = parse_best_json(input1)
    print(f"Test 1 (Largest with diagnosis): {'PASSED' if result1.get('diagnosis') == 'Malaria' else 'FAILED'}")
    if result1.get('diagnosis') != 'Malaria':
        print(f"  Result: {result1}")

    # Test case: Fragment vs full object
    input2 = """
    {"diagnosis": "Unknown"}
    {"diagnosis": "Pneumonia", "extra": "info"}
    """
    result2 = parse_best_json(input2)
    # The second one is longer
    print(f"Test 2 (Longest): {'PASSED' if result2.get('diagnosis') == 'Pneumonia' else 'FAILED'}")

if __name__ == "__main__":
    test_strip_thoughts()
    test_parse_best_json()
