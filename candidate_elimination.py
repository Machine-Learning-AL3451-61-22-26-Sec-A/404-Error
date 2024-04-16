
import streamlit as st
import numpy as np
import pandas as pd

class CandidateElimination:
    def __init__(self, num_attributes, possible_attribute_values):
        self.num_attributes = num_attributes
        self.possible_attribute_values = possible_attribute_values
        self.S = [(['?'] * num_attributes,)]  # most general hypothesis
        self.G = [(['0'] * num_attributes,)]  # most specific hypothesis

    def fit(self, data):
        for instance in data:
            attributes, label = instance[:-1], instance[-1]
            if label == 'yes':  # Positive instance
                self.G = [g for g in self.G if self.is_consistent(g, attributes)]
                self.remove_inconsistent_sets(attributes, self.S)
                self.update_general_hypotheses(attributes)
            else:  # Negative instance
                self.S = [s for s in self.S if self.is_consistent(s, attributes)]
                self.remove_inconsistent_sets(attributes, self.G)
                self.update_specific_hypotheses(attributes)

    def is_consistent(self, hypothesis, instance):
        for h, i in zip(hypothesis, instance):
            if h != '?' and h != i:
                return False
        return True

    def remove_inconsistent_sets(self, instance, hypotheses):
        to_remove = []
        for h in hypotheses:
            if not self.is_consistent(h, instance):
                to_remove.append(h)
        for h in to_remove:
            hypotheses.remove(h)

    def update_general_hypotheses(self, instance):
        new_general_hypotheses = []
        for i, (g, s) in enumerate(zip(self.G[0], instance)):
            if g != '?' and g != s:
                new_hypothesis = list(self.G[0])
                new_hypothesis[i] = '?'
                new_general_hypotheses.append(tuple(new_hypothesis))
        self.G = new_general_hypotheses

    def update_specific_hypotheses(self, instance):
        new_specific_hypotheses = []
        for g in self.G:
            for i, (h, s) in enumerate(zip(g, instance)):
                if h == '?':
                    new_hypothesis = list(g)
                    new_hypothesis[i] = s
                    new_specific_hypotheses.append(tuple(new_hypothesis))
        self.S = new_specific_hypotheses

def main():
    st.title('Candidate Elimination Algorithm')

    st.sidebar.header('Settings')
    num_attributes = st.sidebar.number_input('Number of attributes', min_value=1, value=2)
    possible_attribute_values = st.sidebar.text_input('Possible attribute values (comma-separated)', '0,1')
    possible_attribute_values = possible_attribute_values.split(',')

    st.sidebar.header('Data')
    data = st.sidebar.text_area('Enter data (one instance per line, attributes separated by commas)', '''
0,1,yes
1,1,no
1,0,yes
0,0,no
    ''')
    data = [line.strip().split(',') for line in data.split('\n') if line.strip()]
    
    if st.sidebar.button('Run Candidate Elimination'):
        model = CandidateElimination(num_attributes, possible_attribute_values)
        model.fit(data)

        st.header('Final Hypotheses')
        st.write('S:', model.S)
        st.write('G:', model.G)

if __name__ == '__main__':
    main()

