import streamlit as st

def candidate_elimination(examples):
    specific_h = examples[0][:-1]
    general_h = [["?" for _ in range(len(specific_h))] for _ in range(len(specific_h))]
    
    for example in examples:
        if example[-1] == "Y":
            for i in range(len(specific_h)):
                if example[i] != specific_h[i]:
                    specific_h[i] = "?"
                    general_h[i][i] = "?"
        elif example[-1] == "N":
            for i in range(len(specific_h)):
                if example[i] != specific_h[i]:
                    general_h[i][i] = specific_h[i]
                else:
                    general_h[i][i] = "?"
                    
    return specific_h, [h for h in general_h if h != ["?" for _ in range(len(specific_h))]]

def main():
    st.title("Candidate Elimination Algorithm")
    st.write("This algorithm finds the maximally specific hypothesis and the maximally general hypotheses.")
    
    num_attributes = st.number_input("Number of attributes:", min_value=1, step=1, value=3)
    
    examples = []
    st.write("Enter training examples:")
    for _ in range(st.number_input("Number of examples:", min_value=1, step=1, value=3)):
        example = []
        for i in range(num_attributes + 1):
            if i < num_attributes:
                example.append(st.text_input(f"Attribute {i + 1}:"))
            else:
                example.append(st.selectbox("Label:", options=["Y", "N"]))
        examples.append(example)
    
    if st.button("Run Algorithm"):
        specific, general = candidate_elimination(examples)
        st.write("Maximally Specific Hypothesis:", specific)
        st.write("Maximally General Hypotheses:")
        for h in general:
            st.write(h)

if __name__ == "__main__":
    main()
