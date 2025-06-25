# textstyle.py (updated with Hugging Face)
import os
import streamlit as st
from transformers import pipeline

def text_style_transfer(input_text, target_style="casual"):
    """Transform text to specified style using Hugging Face models"""
    
    # Define style descriptions
    style_descriptions = {
        "professional": "formal business language that is polite, structured, and appropriate for workplace communication",
        "casual": "friendly, relaxed, and conversational tone like talking to a friend",
        "shakespearean": "Elizabethan English in the style of William Shakespeare with 'thee', 'thou', 'verily', etc.",
        "sarcastic": "witty, ironic tone that conveys meaning through clever verbal irony",
        "academic": "scholarly, formal writing style suitable for research papers and academic publications",
        "poetic": "artistic and expressive language with rhythm, imagery, and creative metaphors"
    }

    # Validate style input
    if target_style.lower() not in style_descriptions:
        available_styles = ", ".join(style_descriptions.keys())
        return f"Error: Unsupported style. Choose from: {available_styles}"

    try:
        # Load model - using T5 for text-to-text transformation
        style_model = pipeline("text2text-generation", model="t5-small")
        
        style_desc = style_descriptions[target_style.lower()]
        
        prompt = f"transform text to {target_style} style: {style_desc}\n\n{input_text}"
        
        result = style_model(prompt, max_length=1024, num_beams=2, early_stopping=True)
        transformed_text = result[0]['generated_text']
        
        if not transformed_text or len(transformed_text.strip()) < 3:
            return f"Could not transform to {target_style} style. Original: {input_text}"
        
        return transformed_text
        
    except Exception as e:
        return f"Error during text transformation: {str(e)}. Please try again."

def main():
    st.set_page_config(
        page_title="Text Style Transfer",
        page_icon="🎭",
        layout="wide"
    )
    
    st.title("🎭 Text Style Transfer")
    st.markdown("Transform your text into different styles while preserving the core meaning.")
    
    # Available styles
    styles = ["professional", "casual", "shakespearean", "sarcastic", "academic", "poetic"]
    
    # Input section
    input_text = st.text_area(
        "Enter your text:",
        placeholder="Type or paste your text here...",
        height=150
    )
    
    # Style selection
    target_style = st.selectbox(
        "Select target style:",
        styles
    )
    
    # Transform button with white background
    if st.button("Transform Text", type="primary"):
        if not input_text.strip():
            st.error("Please enter some text to transform!")
            return
            
        with st.spinner(f"Transforming text to {target_style} style..."):
            try:
                result = text_style_transfer(input_text.strip(), target_style)
                
                # Display results
                st.markdown("### Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Original Text:**")
                    st.text_area("Original text display", value=input_text, height=100, disabled=True, label_visibility="hidden")
                
                with col2:
                    st.markdown(f"**{target_style.title()} Style:**")
                    st.text_area("Transformed text display", value=result, height=100, disabled=True, label_visibility="hidden")
                    
            except Exception as e:
                st.error(f"Error during transformation: {str(e)}")

if __name__ == "__main__":
    main()