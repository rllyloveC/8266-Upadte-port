import streamlit as st
import numpy as np
import os
import gc, re, base64
from io import BytesIO
from PIL import Image, ImageEnhance
import easyocr
from llama_cpp.llama_chat_format import Llava15ChatHandler as Qwen2VLChatHandler
import llama_cpp
from llama_cpp import Llama


# ================== CONFIG & SESSION STATE ==================
st.set_page_config(page_title="General AI notepad CODA", layout="wide", page_icon="🤺🤺")

if "shared_problem" not in st.session_state: st.session_state.shared_problem = ""
if "vision_desc" not in st.session_state: st.session_state.vision_desc = ""
if "messages" not in st.session_state: st.session_state.messages = []
if "current_model_path" not in st.session_state: st.session_state.current_model_path = None
if "selected_mode" not in st.session_state: st.session_state.selected_mode = "ให้ Qwen อ่าน"

#==========================================================================


current_file_path = os.path.abspath(__file__)
BASE_DIR = os.path.dirname(current_file_path)
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATHS = {
    "vision": os.path.join(MODEL_DIR, "vision", "Qwen2-VL-7B-Instruct-Q4_K_S.gguf"),
    "vision_projector": os.path.join(MODEL_DIR, "vision", "mmproj-Qwen2-VL-7B-Instruct-f32.gguf"),
    "logic": os.path.join(MODEL_DIR, "logic", "deepseek-r1-distill-qwen-7b-q4_k_m.gguf"),
    "chat": os.path.join(MODEL_DIR, "chat", "qwen2.5-7b-instruct-q4_k_m.gguf") 
}




# ================== SIDEBAR SETTINGS ==================
import streamlit as st
import os


with st.sidebar:
    st.header("🔧 System Settings")
    
    n_gpu = st.slider("GPU Layers (0=CPU)", 0, 50, 0)
    st.session_state.n_gpu = n_gpu
    n_gpu_layers = 0

    app_mode = st.selectbox(
        "เลือกโหมดการทำงาน:",
        ["Qwen2-VL", "Deepseek", "Qwen2.5"]
    )
    mode_to_key = {
        "Qwen2-VL": "vision",
        "Deepseek": "logic",
        "Qwen2.5": "chat"
    }
    selected_key = mode_to_key[app_mode]

def get_single_model(model_key):
    path = MODEL_PATHS[model_key]
    
    if st.session_state.get("current_model_key") != model_key:
        with st.spinner(f"กำลังสลับไปที่ {model_key}... กรุณารอสักครู่"):
            if "llm" in st.session_state:
                del st.session_state.llm
            if "vision_chat_handler" in st.session_state:
                del st.session_state.vision_chat_handler
            gc.collect()
            
            if not os.path.exists(path):
                st.error(f"หาไฟล์ไม่พบที่: {path}")
                return None

            chat_handler = None
            if model_key == "vision":
                from llama_cpp.llama_chat_format import Llava15ChatHandler # Fallback
                chat_handler = Llava15ChatHandler(
                    clip_model_path=MODEL_PATHS["vision_projector"]
                )
                st.session_state.vision_chat_handler = chat_handler

            st.session_state.llm = Llama(
                model_path=path,
                chat_handler=chat_handler,
                n_ctx=1024,
                n_gpu_layers=st.session_state.get("n_gpu", 0)
            )
            st.session_state.current_model_key = model_key
            st.success(f"โหลด {model_key} สำเร็จ!")

    st.divider()        
    return st.session_state.llm

    if st.button("🗑️ Clear All Memory"):
        st.session_state.messages = []
        st.session_state.vision_desc = ""
        st.session_state.shared_problem = ""
        st.rerun()

    st.subheader("Model Status Check")

    all_models_ok = True
    
    if 'MODEL_PATHS' in globals():
        for name, path in MODEL_PATHS.items():
            if os.path.exists(path):
                st.success(f"✅ {name}: Found")
            else:
                st.error(f"❌ {name}: Not Found")
                st.caption(f"Path: {path}")
                all_models_ok = False
        
        if not all_models_ok:
            st.warning("กรุณาตรวจสอบ Path ของโมเดลก่อนเริ่มใช้งาน")
    else:
        st.info("No models configured to check.")

# ================== CORE FUNCTIONS ==================

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['th', 'en'], gpu=False)


ocr = load_ocr()


def get_model(model_key):
    path = MODEL_PATHS[model_key]
    if st.session_state.current_model_path != path:
        if "llm" in st.session_state:
            del st.session_state.llm
        if "vision_chat_handler" in st.session_state:
            del st.session_state.vision_chat_handler

        gc.collect()
        st.warning(f"🔄 Switching to {model_key.upper()} Model...")

        chat_handler = None


        if model_key == "vision":
            try:
                chat_handler = chat_handler = Qwen2VLChatHandler(clip_model_path=MODEL_PATHS["vision_projector"],
                            verbose=False
                               )
                st.session_state.vision_chat_handler = chat_handler
            except AttributeError:
                st.error(
                    "ไม่พบ Qwen2VLChatHandler "
                    "กรุณาตรวจสอบเวอร์ชัน llama-cpp-python"
                )
                chat_handler = None

        st.session_state.llm = Llama(
            model_path=path,
            chat_handler=chat_handler,
            n_ctx=4096,
            n_gpu_layers=st.session_state.n_gpu,
            logits_all=True if model_key == "vision" else False,
            verbose=False
        )
        st.session_state.current_model_path = path
    return st.session_state.llm



def extract_text_from_image(pil_image):
    img = np.array(pil_image)
    results = ocr.readtext(img, detail=0, paragraph=True)
    text = "\n".join(results)
    return text.strip()
def describe_image_meta(pil_image):
    w, h = pil_image.size
    return f"""
Image Metadata:
- Resolution: {w} x {h} pixels
- Color Mode: {pil_image.mode}
"""
        
def encode_image_to_base64(pil_image):
    if pil_image.mode in ("RGBA", "P"): pil_image = pil_image.convert("RGB")
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def image_quality_score(img):
    gray = np.mean(img, axis=2)
    sharp = np.var(np.gradient(gray))
    glare = np.mean(gray > 240)
    return sharp - glare * 1000

def needs_calculation(text):
    keywords = ["ลองคิด", "หาให้หน่อย", "ทำให้หน่อย", "ทำให้ดู", "ลองทำให้หน่อย"]
    return any(k in text.lower() for k in keywords) or bool(re.search(r"[\+\-\*/=\^]", text))

def auto_detect_mode(text: str) -> str:
    t = text.lower().strip()
    
    # Physics Keywords (เพิ่มความหลากหลาย)
    physics_keywords = ["แรง", "ความเร็ว", "มวล", "force", "velocity", "mass", "acceleration", "m/s"]
    # Math Keywords
    math_keywords = ["สมการ", "integral", "derivative", "matrix", "จงหาค่า", "limit"]
    # Science Keywords
    science_keywords = ["ปฏิกิริยา", "โมเลกุล", "สาร", "ph", "reaction", "acid", "base"]

    if any(k in t for k in physics_keywords):
        return "Forced Physics" 
    if any(k in t for k in math_keywords) or bool(re.search(r"\\frac|\\int|\\sum", t)):
        return "Forced Math"
    if any(k in t for k in science_keywords):
        return "General Science"
        
    return "ให้ Qwen อ่าน"

# --- PROMPT SETS ---------------------------------------------------
SOLVER_PROMPT_SET = {
    "ให้ Qwen อ่าน": """Role: Professional Data Analyst & Technical Consultant.
Task: Provide a clear explanation or summary based on the provided 'Vision Context'.
Instructions:
- Analyze the information extracted from the image (tables, text, or general scenes).
- If it's a general question, provide a helpful and concise response.
- If it contains data/statistics, highlight key trends and insights.
- ALWAYS respond in Thai.
- Maintain a helpful and professional tone.
""",
    "Forced Physics": """Role: Expert Physics Professor.
Task: Solve the physics problem provided in the 'Vision Context' step-by-step.
Instructions:
- Analyze the physical scenario and identified variables from the context.
- Identify: 1) Given parameters, 2) Unknowns to solve, and 3) Relevant physical laws/formulas.
- Perform calculations step-by-step.
- Use LaTeX for ALL mathematical expressions, variables, and units (e.g., $v = u + at$, $9.8 \text{ m/s}^2$).
- Verify units and physical consistency in the final answer.
- Reasoning process can be internal, but the final output MUST be in Thai.
""",
    "Forced Math": """Role: Master of Mathematics.
Task: Solve the mathematical problem or prove the statement in the 'Vision Context'.
Instructions:
- Interpret the mathematical structure and notation from the provided LaTeX text.
- Define necessary theorems, axioms, or properties being used.
- Execute calculations or logical proofs line-by-line with absolute precision.
- Use LaTeX for ALL mathematical symbols, equations, and expressions without exception.
- State the final result clearly with its domain or constraints.
- Output the explanation and steps in Thai language.""",
    "General Science": """Role: Multidisciplinary Science Expert (Chemistry/Biology/General Science).
Task: Explain scientific phenomena or solve science-related problems in the 'Vision Context'.
Instructions:
- Synthesize the visual evidence to identify the scientific domain.
- For Chemistry: Provide balanced chemical equations using LaTeX, including states of matter and reaction conditions.
- For Biology/General Science: Explain the cause-and-effect relationships based on scientific principles.
- If there's experimental data or graphs, provide an objective interpretation.
- Use LaTeX for all constants, formulas, and chemical symbols.
- Respond and explain in Thai.
"""
}

PROMPT_SET = {
    "ให้ Qwen อ่าน": """
Role: Vision-Language Parsing Expert.
Task: Carefully read and extract information from the given image, which may include graphs, tables, charts, or annotated visuals.
Key Instructions (STRICT):
- Do NOT analyze trends or draw conclusions.
- Do NOT interpret meaning beyond what is explicitly shown.
- Extract only visible information.
Key Elements to Extract:
- Titles, labels, legends, and captions.
- Axes names, units, scales, and tick values.
- Table headers, rows, columns, and numeric values.
- Symbols, colors, markers, and annotations.
- Any text embedded in the image.
Output Format:
- Image Type (graph / table / diagram / mixed)(if present)
- Extracted Text (verbatim)
- Graph/Table Structure
- Numerical Data (as listed)
- Notes on unclear or unreadable parts ([??])
""",
    "Forced Physic": """
Role: Universal Physics Vision-Language Parsing Expert.
Task: Extract and transcribe all visible information from the given physics problem image.
STRICT RULES:
- Do NOT solve the problem.
- Do NOT explain or interpret physics.
- Do NOT choose formulas or laws.
- Extract only what is explicitly shown.
Language & Format Rules:
- Write all text in clear academic English.
- Write all mathematical expressions in clean LaTeX.
- Preserve symbols, subscripts, superscripts, vectors, and notation exactly.
- If a symbol is unclear, mark it as \\text{[??]}.
Elements to Extract:
- Problem statements (verbatim).
- All equations and expressions (LaTeX).
- Physical quantities, symbols, and units.
- Diagrams, graphs, tables (described objectively).
- Boundary conditions, initial conditions, constraints.
- Questions asked.
- Multiple-choice options (if present).
Output Format:
- Detected Physics Domains (visual evidence only)
- Problem Statement
- Physical Quantities and Units
- Equations and Expressions
- Diagram / Graph / Table Description
- Given Conditions and Constraints
- Questions Asked
- Unclear or Ambiguous Elements
Do NOT include explanations or solutions.
""",
    "Forced Math": """
Role: Mathematical OCR and Symbol Parsing Expert.
Task: Transcribe a mathematics problem from the image with maximum symbol and variable fidelity.
STRICT RULES (ULTRA-STRICT):
- Do NOT solve or simplify.
- Do NOT rephrase the problem.
- Do NOT assume mathematical intent.
- Preserve all notation exactly as shown.
Language & Format Rules:
- Write all text in precise academic English.
- Represent ALL mathematics using clean LaTeX.
- Preserve:
  - Variable names (case-sensitive).
  - Subscripts and superscripts.
  - Parentheses, brackets, and braces.
  - Fractions, radicals, sums, integrals, limits.
  - Matrices, vectors, and piecewise definitions.
- If any symbol or variable is unclear, mark it as \\text{[UNCLEAR]}.
Key Elements to Extract:
- Full problem statement (verbatim).
- Definitions of variables or functions.
- All equations, inequalities, and expressions.
- Constraints, domains, and conditions.
- Diagrams or graphs (described textually).
- Final question(s) exactly as written.
Output Format:
- Problem Statement
- Variable and Symbol List (as shown)
- Equations and Expressions (LaTeX)
- Conditions and Constraints
- Diagram / Graph Description
- Questions Asked
- Unclear or Ambiguous Elements
Do NOT include explanations or solutions.
""",
    "General Science": """
Role: Scientific Vision-Language Parsing Expert.
Task: Extract and transcribe all visible information from a science problem image, including physics and chemistry content.
STRICT RULES (NO REASONING):
- Do NOT solve, explain, or interpret the problem.
- Do NOT infer chemical reactions or physical laws.
- Do NOT balance equations or simplify expressions.
- Extract only what is explicitly visible.
Language & Format Rules:
- Write all text in clear academic English.
- Write all mathematical and chemical expressions in clean LaTeX.
- Preserve subscripts, superscripts, charges, states, arrows, and special symbols exactly.
- If any character or symbol is unclear, mark it as \\text{[UNCLEAR]}.
Key Elements to Extract:
- Problem statements (verbatim).
- Chemical formulas, reactions, and equations.
- Physical equations and expressions.
- Symbols, constants, and units.
- Tables, graphs, diagrams, and annotations.
- Experimental conditions (temperature, pressure, volume, concentration).
- Given values and constraints.
- Questions asked.
- Multiple-choice options (if present).
Output Format:
- Detected Science Domains (visual evidence only)
- Problem Statement
- Chemical Species / Physical Quantities and Units
- Equations, Reactions, and Expressions
- Diagram / Graph / Table Description (objective)
- Given Conditions and Constraints
- Questions Asked
- Unclear or Ambiguous Elements
- Do NOT include explanations or solutions.
"""
}

# ================== UI TABS ===========================================
tab1, tab2, tab3 = st.tabs(["Vision Scanner", "Performance process", "General chat"])

# --- TAB 1: VISION ----=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
with tab1:
    st.header("Text and Vision Processor")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        source = st.radio("Input Source:", ["📂 Upload Files", "📸 Camera"], horizontal=True)
        # UI Selection
        selected_mode_ui = st.selectbox(
            "Mode:", 
            list(PROMPT_SET.keys()),
            index=list(PROMPT_SET.keys()).index(st.session_state.selected_mode),
            key="selected_mode_ui"
        )
        # Update session state logic
        st.session_state.selected_mode = selected_mode_ui

    best_image_pil = None
    with col2:
        if source == "📸 Camera":
            cam_file = st.camera_input("Capture")
            if cam_file: best_image_pil = Image.open(cam_file)
        else:
            files = st.file_uploader("Upload", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
            if files:
                scored = []
                for f in files:
                    p = Image.open(f)
                    s = image_quality_score(np.array(p))
                    scored.append((s, p))
                best_image_pil = max(scored, key=lambda x: x[0])[1]
                st.image(best_image_pil, caption="Best Image Selected", width=400)

    if best_image_pil:
        if st.button("🚀 Analyze Image", type="primary"):
            llm = get_model("vision")
            
            enhancer = ImageEnhance.Contrast(best_image_pil)
            proc_img = enhancer.enhance(1.4)
            
            with st.spinner("Processing OCR & Vision..."):
                ocr_text = extract_text_from_image(proc_img)
                base64_img = encode_image_to_base64(proc_img)
                current_system_prompt = PROMPT_SET[st.session_state.selected_mode]
                
                # Fixed indentation and f-string
                ocr_prompt = (
                              "OCR Reference (may contain errors):\n"
                              f"{ocr_text}\n\n"
                              "IMPORTANT: Prioritize visual content from the image over OCR text."
                                )
        

                res = llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": current_system_prompt},
                        {"role": "user", "content": [
                            {"type": "text", "text": ocr_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                        ]}
                    ],
                    max_tokens=1024,
                    temperature=0.2
                )

                desc = res["choices"][0]["message"]["content"]
                
                # Auto Detect Mode Logic
                if st.session_state.selected_mode == "ให้ Qwen อ่าน":
                    detected_mode = auto_detect_mode(ocr_text + "\n" + desc)
                    st.session_state.selected_mode = detected_mode
                    st.toast(f"Auto-detected mode: {detected_mode}")

                st.session_state.vision_desc = desc
                st.session_state.shared_problem = desc
                st.rerun()

    if st.session_state.vision_desc:
        with st.expander("Analysis Result", expanded=True):
            st.markdown(st.session_state.vision_desc)

# --- TAB 2: LOGIC ---
with tab2:
    st.header("Logically thinking")
    problem = st.text_area("Problem Statement:", st.session_state.shared_problem, height=200)
    
    if st.button("🧠 Start Reasoning", disabled=not problem):
        llm = get_model("logic")
        base_logic_prompt = SOLVER_PROMPT_SET.get(st.session_state.selected_mode, "You are an expert.")
        
        # ใช้ Chat Format แทน Plain Text เพื่อประสิทธิภาพที่ดีกว่าของ DeepSeek/Qwen
        messages = [
            {"role": "system", "content": base_logic_prompt + "\nAnswer in Thai language."},
            {"role": "user", "content": f"Vision Context:\n{st.session_state.vision_desc}\n\nProblem:\n{problem}"}
        ]
        
        with st.spinner("DeepSeek is thinking..."):
            placeholder = st.empty()
            full_response = ""
            
            # ใช้ create_chat_completion stream
            stream = llm.create_chat_completion(messages=messages, max_tokens=2048, stream=True)
            
            for chunk in stream:
                if "content" in chunk["choices"][0]["delta"]:
                    content = chunk["choices"][0]["delta"]["content"]
                    full_response += content
                    placeholder.markdown(full_response + "▌")
            
            placeholder.markdown(full_response)

# --- TAB 3: CHAT --------------------------------------------------
with tab3:
    st.header("General Chat Assistant")
    
    # Display History
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if p := st.chat_input("นั่งคุยเรื่องราวอะไรดี"):
        st.session_state.messages.append({"role": "user", "content": p})
        with st.chat_message("user"): st.markdown(p)
        
        is_calc = needs_calculation(p)
        model_type = "logic" if is_calc else "chat"
        llm = get_model(model_type)
        
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""
            
            # Construct messages properly for Chat Completion
            # เอา history ล่าสุดมาใส่ context
            chat_history = st.session_state.messages[-5:] # เอาแค่ 5 ข้อความล่าสุดเพื่อประหยัด Context
            
            if is_calc:
                sys_msg = {"role": "system", "content": "You are a helpful assistant capable of logical reasoning and calculation. Answer in Thai."}
                input_msgs = [sys_msg] + chat_history
            else:
                sys_msg = {"role": "system", "content": "You are a helpful assistant. Answer in Thai."}
                input_msgs = [sys_msg] + chat_history

            stream = llm.create_chat_completion(messages=input_msgs, max_tokens=1024, stream=True)
            
            for chunk in stream:
                if "content" in chunk["choices"][0]["delta"]:
                    content = chunk["choices"][0]["delta"]["content"]
                    full_response += content
                    placeholder.markdown(full_response + "▌")
            
            placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
