

## Assignment Overview

This assignment explores the application of Q-learning, a reinforcement learning technique, to a character-level text generation model. The goal is to train the model to generate text that aligns with specific criteria, influenced by a reward function. The assignment compares text generation before and after Q-learning and investigates the effects of different reward criteria and weight adjustment strategies.

## Questions Specific to This Assignment

### 1. Criteria Function: `lambda x: -len(x)`

*   **Question:** With a criteria function of `lambda x: -len(x)`, did it encourage the model to generate shorter texts? Why or why not?
*   **Answer:** Yes, this function is designed to reward shorter texts. The reward is the negative of the text's length. Therefore, shorter texts (smaller length `len(x)`) result in a less negative (i.e., higher) reward.
*   **Example:**
    *   **BEFORE Q-learning (with this criteria):**
        *   Prompt: `stranger`
        *   Length/Score: 47.7 (This seems to be an average length from a previous run, not the reward value itself)
        *   Generated: `stranger I know everything about`
    *   **AFTER Q-learning (with this criteria):**
        *   Prompt: `stranger`
        *   Length/Score: 42.6
        *   Generated: `'stranger I knows that they believe it's nice\n'`
    *   **Observation:** The generated text after Q-learning is indeed shorter, and the average length/score metric decreased, indicating the model learned to prefer shorter sequences.

### 2. Custom Criteria Function (Rewarding Specific Words)

*   **Task:** Try out a criteria function of your choice. The example implemented rewards the presence of positive words.
*   **Implementation:**
    *   A custom reward function was designed: add 5 to a counter if any of the words "happy," "nice," or "love" occur in the generated text. This count is then used as the reward in the Q-learning update.
    *   The Q-learning update rule snippet provided:
        ```python
        # (Simplified, assuming word_counts is based on the generated text for the current state-action)
        # for word, count in word_counts.items(): # This part seems more related to the training text processing
        # The core Q-learning update for the next_letter based on a reward (e.g., presence of "happy", "nice", "love")
        # if next_letter in window_freq:
        #     window_freq[next_letter] = (1 - self.alpha) * window_freq[next_letter] + self.alpha * (reward + self.gamma * max(next_window_freq.values()))
        # else:
        #     window_freq[next_letter] = self.alpha * (reward + self.gamma * max(next_window_freq.values(), default=0))
        ```
        The document shows a snippet related to updating `window_freq` based on `count` which seems to be from the n-gram model rather than the direct Q-learning update with the custom reward. The description implies the *rewarded count* (e.g., 5 if "happy" is present) is used in the Q-function update.
*   **Outcome:**
    *   The model was able to generate text including one of the target words (happy, nice, love) "once in 10 times."
*   **Example:**
    *   **BEFORE Q-learning (with this criteria):**
        *   Prompt: `stranger`
        *   Generated: `stranger I know everythings I did to try and win your loved you can't even remember what made me lose all the day that I am the storm cl`
    *   **AFTER Q-learning (with this criteria, showing varied outputs):**
        *   Prompt: `stranger`
        *   Generated (example containing "love"): `stranger I know ever doin' and now I'll love`
        *   Other generated examples: `stranger I know`, `stranger I know everybody told me it would burn out of my life`, etc.
    *   **Observation:** The model showed some tendency to incorporate the rewarded words, though not consistently in every generation.

### 3. Softmax for Negative Q-Values and Longer Songs

*   **Question:** In Homework 1, the character n-gram model used frequencies as weights. For this assignment (initially referred to as Homework 5, likely a typo for Assignment 4), Q-learning can result in negative Q-values. Using the softmax of weights resulted in longer generated songs. Why?
*   **Answer:**
    *   The softmax function converts a vector of values into a probability distribution by exponentiating each value and then normalizing by the sum of all exponentiated values.
    *   **Exponentiation Effect:** Softmax amplifies differences between values. Higher Q-values (even if originally negative but less so after potential transformations or relative to others) become significantly more probable after exponentiation compared to lower Q-values.
    *   **Skewed Distribution:** This leads to a more skewed probability distribution where certain characters (those with higher Q-values after transformation for softmax) are much more likely to be chosen.
    *   **Repetitive Sequences & Longer Text:** If the model becomes very confident in certain sequences due to these amplified probabilities, it's more likely to generate those sequences repeatedly or extend them, leading to longer texts.
*   **Example (illustrating length change, though this example uses the `-len(x)` criteria which shortens text):**
    *   **BEFORE Softmax (or Q-learning with different reward):**
        *   Prompt: `stranger`
        *   Score/Length: 50.1
        *   Generated: `stranger I know everybody told me it was half myself without you, I'll everything about`
    *   **AFTER (presumably Q-learning with `-len(x)` reward, showing the intended effect of that specific reward):**
        *   Prompt: `stranger`
        *   Score/Length: 64.9 (This value seems to contradict the shortening effect of `-len(x)`. The generated text is `'stranger I know\n'`, which *is* shorter. The score might be mislabeled or from a different context/run.)
        *   Generated: `'stranger I know\n'`
    *   **Note:** The example provided here under the "softmax" section actually demonstrates the effect of the `-len(x)` reward (shorter text), not directly the lengthening effect of softmax if the underlying Q-values promoted longer sequences. The explanation for why softmax *could* lead to longer texts (by amplifying preferences for certain characters/sequences) is the key takeaway.

### 4. Weight Adjustment: Subtracting Minimum and Adding 1

*   **Question:** We also tried subtracting the minimum Q-value from all Q-values in a state and then adding 1 to each. Why add 1 instead of just subtracting the minimum?
*   **Answer:** By adding 1 after subtracting the minimum, all Q-values (or weights derived from them) become **strictly positive** (greater than zero). This is often important for subsequent probability calculations (like softmax, or if using them as direct probabilities/frequencies) where zero or negative inputs might be problematic or undefined. Subtracting the minimum alone would make the smallest value zero, which could still cause issues (e.g., zero probability).

### 5. Controlled Comparison: Same Prompt Before and After Q-Learning

*   **Question:** Why is the difference in the length of generated songs (as a result of Q-learning) more dramatic if we test the same prompt before and after Q-learning, compared to using different prompts?
*   **Answer:**
    *   **Controlled Comparison:** Using the same prompt provides a controlled environment to isolate and observe the effects of the Q-learning process itself. It removes variability that would be introduced by different starting contexts.
    *   **Reinforced Patterns:** The learning process (Q-learning updates) may have specifically reinforced or altered the probabilities of certain character transitions or patterns that are particularly relevant to that *specific prompt*. When tested with the same prompt, these learned changes are directly engaged, leading to a more noticeable impact on the generated text's characteristics (like length).


*   **Resources Used:**
    *   Course slides
    *   Olivia Rodrigo's songs dataset: [Kaggle - Olivia Rodrigo Lyrics](https://www.kaggle.com/datasets/mehaksingal/olivia-rodrigo-lyrics-datasetl)
    *   Used the lyric "stranger" as training text.
*   **Most Difficult Part:**
    *   Designing the reward/criteria function: It should consider the generated text from the first character up to the current window + next letter.
    *   Timing of weight conversion: Negative weights should be converted (e.g., for softmax) after each Q-learning iteration (total 30 iterations).
    *   Difficulty with interactive reward: Designing a criteria function that asks the user for a score for generated text is challenging because Q-learning updates weights per character generation (based on window + next character), which would require frequent user input if the reward is tied to the entire generated text up to that point. The Q-learning process updates weights for state-action pairs (window -> next character), not for the entire generated sequence at once.
*   **Most Rewarding Part:**
    *   Being able to try the Q-learning model and compare it with the n-gram model.
*   **What was Learned:**
    *   How to design a reward based on how weights in the n-gram model are calculated. The reward should be relative to counts and easy to update when the window slides.

