import numpy as np
from collections import OrderedDict


class HMM:

    def __init__(self):
        self.states = set()
        self.observations = set()
        self.transition_probs = {}
        self.emission_probs = {}
        self.initial_probs = {}

    def train(self, corpus):
        for sentence in corpus:
            prev_state = None
            for word, state in sentence:
                self.states.add(state)
                self.observations.add(word)
                if prev_state is None:
                    self.initial_probs[state] = self.initial_probs.get(
                        state, 0) + 1
                else:
                    self.transition_probs[(prev_state,
                                           state)] = self.transition_probs.get(
                                               (prev_state, state), 0) + 1
                self.emission_probs[(state, word)] = self.emission_probs.get(
                    (state, word), 0) + 1
                prev_state = state

        # Normalize probabilities
        self.initial_probs = {
            state: count / len(corpus)
            for state, count in self.initial_probs.items()
        }
        for state in self.states:
            total_transition_count = sum(
                self.transition_probs.get((state, next_state), 0)
                for next_state in self.states)
            # Handle the case where total_transition_count is zero
            if total_transition_count == 0:
                # Assign equal probabilities for all transitions
                self.transition_probs.update({(state, next_state):
                                              1 / len(self.states)
                                              for next_state in self.states})
            else:
                self.transition_probs.update({
                    (state, next_state):
                    np.floor(count / total_transition_count)
                    for (state,
                         next_state), count in self.transition_probs.items()
                })
            total_emission_count = sum(
                self.emission_probs.get((state, word), 0)
                for word in self.observations)
            self.emission_probs.update({
                (state, word): count / total_emission_count
                for (state, word), count in self.emission_probs.items()
            })

        # self.print_transition_matrix()
        # print(self.initial_probs)
        # print(self.emission_probs)

    def predict(self, sentence):
        T = len(sentence)
        V = np.zeros((T, len(self.states)))
        backpointers = np.zeros((T, len(self.states)), dtype=int)

        # Initialization
        # state_indices = {state: i for i, state in enumerate(self.states)}
        state_indices = OrderedDict(
            (state, i) for i, state in enumerate(self.states))

        # Initialize V matrix using initial probabilities
        for state, prob in self.initial_probs.items():
            V[0, state_indices[state]] = prob
        # print("After initialization, V matrix:\n", V)

        # Forward pass
        for t in range(1, T):
            for s, next_state in self.transition_probs.keys():
                for next_state_index, next_state_name in enumerate(
                        self.states):
                    transition_prob = self.transition_probs.get(
                        (s, next_state_name), 0)
                    emission_prob = self.emission_probs.get(
                        (next_state_name, sentence[t]), 0)
                    new_prob = V[
                        t - 1,
                        state_indices[s]] * transition_prob * emission_prob
                    if new_prob > V[t, next_state_index]:
                        V[t, next_state_index] = new_prob
                        backpointers[t, next_state_index] = state_indices[s]
            # print(f"After time step {t}, V matrix:\n", V)

        # Backtracking
        best_path_indices = [np.argmax(V[-1])]
        for t in range(T - 1, 0, -1):
            best_path_indices.append(backpointers[t, best_path_indices[-1]])
        best_path_indices.reverse()
        # print("Best path indices:", best_path_indices)

        # Convert indices to POS strings
        best_path = [list(self.states)[index] for index in best_path_indices]
        # print("Predicted POS tags:", best_path)

        return best_path

    def print_transition_matrix(self):
        transition_probs = self.transition_probs
        # Extract unique POS tags
        pos_tags = set()
        for transition in transition_probs:
            pos_tags.update(transition)

        # Sort POS tags
        pos_tags = sorted(pos_tags)

        # Initialize matrix with zeros
        matrix = [[0.0] * len(pos_tags) for _ in range(len(pos_tags))]

        # Populate matrix with probabilities
        for i, tag1 in enumerate(pos_tags):
            for j, tag2 in enumerate(pos_tags):
                prob = transition_probs.get((tag1, tag2), 0.0)
                matrix[i][j] = prob

        # Print matrix
        print("Transition Matrix:")
        for i, tag1 in enumerate(pos_tags):
            print(f"{tag1}:", end=" ")
            for j, tag2 in enumerate(pos_tags):
                print(f"{matrix[i][j]:.2f}", end=" ")
            print()


# Simple Python generator to append the list to itself
def expand_dataset(corpus, num_times):
    for _ in range(num_times):
        for sentence in corpus:
            yield sentence


def evaluate_hmm(hmm, test_corpus, observation_index_func):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for sentence in test_corpus:
        # Extract words and true POS tags from the test sentence
        words = [word for word, _ in sentence]
        true_tags = [tag for _, tag in sentence]

        # Predict POS tags for the test sentence
        predicted_tags = hmm.predict(words, observation_index_func)

        # Compare predicted tags with true tags
        for pred_tag, true_tag in zip(predicted_tags, true_tags):
            if pred_tag == true_tag:
                true_positives += 1
            else:
                if pred_tag not in true_tags:
                    false_positives += 1
                else:
                    false_negatives += 1

    # Compute precision, recall, and F1-score
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)

    # Compute accuracy
    accuracy = true_positives / sum(len(sentence) for sentence in test_corpus)

    # accuracy, precision, recall, f1_score = evaluate_hmm(hmm_pos, test_corpus_pos, observation_index_func_pos)

    # Print the performance metrics
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1_score)

    return accuracy, precision, recall, f1_score


def main(corpus=None):
    if corpus is None:
        corpus = [[("The", "DET"), ("dog", "NOUN"), ("barks", "VERB")],
                  [("A", "DET"), ("cat", "NOUN"), ("meows", "VERB")],
                  [("Some", "DET"), ("birds", "NOUN"), ("sing", "VERB")],
                  [("An", "DET"), ("elephant", "NOUN"), ("trumpets", "VERB")],
                  [("Many", "DET"), ("fish", "NOUN"), ("swim", "VERB")],
                  [("The", "DET"), ("monkey", "NOUN"), ("climbs", "VERB")],
                  [("A", "DET"), ("lion", "NOUN"), ("roars", "VERB")],
                  [("Several", "DET"), ("horses", "NOUN"), ("gallop", "VERB")]]

    # Train
    hmm = HMM()
    hmm.train(corpus)

    # Test
    sentence = ["The", "cat", "runs"]  # Example test sentence
    sentence = ["The", 'cat', 'barks']
    predicted_tags = hmm.predict(sentence)
    print('test sentence: ', sentence)
    print('predicted POS tags: ', predicted_tags)
    return hmm


if __name__ == '__main__':
    main()
