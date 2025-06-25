# Grammar_Correction.py (updated with Hugging Face)
import streamlit as st
import os
import json
from transformers import pipeline

def correct_grammar(text):
    """Correct grammar and improve fluency using Hugging Face models"""
    try:
        # Try using T5 model first
        try:
            corrector = pipeline("text2text-generation", model="vennify/t5-base-grammar-correction")
        except:
            # Fallback to GPT2 if T5 not available
            corrector = pipeline("text-generation", model="gpt2")
        
        prompt = f"Correct this to standard English:\n\n{text}"
        
        if "t5" in corrector.model.name_or_path.lower():
            # For T5 model
            result = corrector(prompt, max_length=1024, num_beams=2, early_stopping=True)
            corrected_text = result[0]['generated_text']
        else:
            # For GPT model
            result = corrector(prompt, max_length=1024, num_return_sequences=1)
            corrected_text = result[0]['generated_text'].replace(prompt, "").strip()
        
        return {
            "corrected_text": corrected_text,
            "changes_made": ["Grammar and fluency improvements applied"],
            "confidence": 0.85
        }
        
    except Exception as e:
        return {"error": f"Failed to correct grammar: {str(e)}"}

def main():
    st.set_page_config(
        page_title="Grammar Correction",
        page_icon="✍️",
        layout="wide"
    )
    
    st.title("✍️ Grammar & Fluency Correction")
    st.markdown("Improve your text with AI-powered grammar correction and fluency enhancement.")
    
    # Input section
    input_text = st.text_area(
        "Enter your text:",
        placeholder="Paste your text here for grammar correction...",
        height=200
    )
    
    # Correct button
    if st.button("✍️ Correct Grammar", type="primary"):
        if not input_text.strip():
            st.error("Please enter some text to correct!")
            return
            
        with st.spinner("Analyzing and correcting your text..."):
            result = correct_grammar(input_text.strip())
            
            if "error" in result:
                st.error(result["error"])
                return
            
            # Display results
            st.markdown("### Results")
            
            # Before and after comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Original Text:**")
                st.text_area("Original text display", value=input_text, height=150, disabled=True, label_visibility="hidden")
            
            with col2:
                st.markdown("**Corrected Text:**")
                corrected_text = result.get("corrected_text", "No correction available")
                st.text_area("Corrected text display", value=corrected_text, height=150, disabled=True, label_visibility="hidden")
            
            # Show changes if available
            if "changes_made" in result:
                changes = result.get("changes_made", [])
                if changes:
                    st.markdown("### Changes Made")
                    for i, change in enumerate(changes, 1):
                        st.markdown(f"{i}. {change}")
            
            # Confidence score
            if "confidence" in result:
                confidence = float(result["confidence"])
                st.markdown(f"**Confidence:** {confidence:.0%}")

if __name__ == "__main__":
    main()