eval_conclusion_withthinktag = """### Task: Evaluate the Conclusion of an LLM-generated response based on a structured benchmark.

#### **Benchmark Structure:**
1. **Query:** A textual question about a point cloud.
2. **Reference Answer:** 
   - **<think>...</think>**: Structured reasoning (e.g., object recognition, functional reasoning, interaction prediction).
   - **Conclusion:** A brief summary outside `<think>`.
3. **LLM-generated Response:**  
   - **<think>...</think>**: Model’s reasoning process.
   - **Conclusion:** The part to evaluate.

🚫 **Do NOT assess `<think>`—evaluate ONLY the Conclusion.**  

---

### **Evaluation Criteria (1-5 scale):**

#### **1. Truthfulness**  
**Does the Conclusion accurately reflect the Reference Answer’s Conclusion?**  

| Score | Description |
|---|---|
| **5** | Perfect match—no incorrect information, even if `<think>` has errors. |
| **4** | Mostly correct—minor wording differences or slight detail omissions. |
| **3** | Partially correct—one significant mistake (e.g., wrong classification, function, or interaction). |
| **2** | Largely incorrect—multiple errors in reasoning or classification. |
| **1** | Completely incorrect—unrecognizable or propagates all `<think>` errors. |

#### **2. Completeness**  
**Does the Conclusion retain critical reasoning layers from `<think>` (object recognition, function, interaction)?**  

| Score | Description |
|---|---|
| **5** | Fully retains all key information—nothing omitted. |
| **4** | Mostly complete—minor omissions (e.g., one missing interaction type). |
| **3** | Omits multiple key points—e.g., missing function or interaction reasoning. |
| **2** | Retains minimal useful content—most reasoning lost. |
| **1** | Extracts no meaningful insights—little to no value. |

---

### **Instructions:**  
- Compare the **LLM’s Conclusion** ONLY against the **Reference Answer’s Conclusion**.  
- **Ignore `<think>` when scoring.**  

### **Expected JSON Output:**
```json
{
  "Truthfulness": {
    "Score": "<1-5>",
    "Explanation": "<Brief justification>"
  },
  "Completeness": {
    "Score": "<1-5>",
    "Explanation": "<Brief justification>"
  }
}
```

---

**Textual Query:**  
[Insert Query]  

**Reference Answer:**  
[Insert Reference Answer]  

**LLM-generated Response:**  
[Insert LLM Response]
"""

eval_cot_withthinktag = """### Task: Evaluate the **reasoning process** in the `<think>` section of an LLM-generated response.

#### **Benchmark Structure:**
1. **Query:** A textual question about a point cloud.
2. **Reference Answer:**  
   - **<think>...</think>**: The structured reasoning process (**CoT reasoning**).  
   - **Conclusion:** A brief summary (not relevant for evaluation).  
3. **LLM-generated Response:**  
   - **<think>...</think>**: The reasoning process to be evaluated.  
   - **Conclusion:** Ignore this section.  

🚫 **Do NOT assess the Conclusion—evaluate ONLY `<think>` reasoning.**

---

### **Evaluation Criteria (1-5 scale):**

#### **1. Object Recognition**  
**Does the model correctly identify the object and its features?**  

| Score | Description |
|---|---|
| **5** | Object and features are correctly identified with precise details. |
| **4** | Correct object category, but some minor details are missing or slightly inaccurate. |
| **3** | Partially correct identification, but some confusion exists. |
| **2** | Mostly incorrect, with only minor accurate elements. |
| **1** | Entirely misidentified or irrelevant to the query. |

#### **2. Functional Reasoning**  
**Does the model accurately infer the function of the object?**  

| Score | Description |
|---|---|
| **5** | Fully correct inference with strong supporting logic. |
| **4** | Mostly correct, but missing some details or clarity. |
| **3** | Some reasonable aspects, but lacks completeness. |
| **2** | Mostly incorrect, though some logical elements exist. |
| **1** | Entirely incorrect or nonsensical. |

#### **3. Interaction Prediction**  
**Does the model provide a logical and coherent description of human-object interaction?**  

| Score | Description |
|---|---|
| **5** | Highly logical and natural interaction, well-described. |
| **4** | Mostly correct, but lacking some finer details. |
| **3** | Some reasonable interactions, but others unclear or missing. |
| **2** | Mostly incorrect or unrealistic. |
| **1** | Completely wrong or nonsensical. |

---

### **Evaluation Process:**
1. **Analyze** the Query and Reference Answer.  
2. **Focus only** on the `<think>` section of the LLM’s response.  
3. **Compare** its reasoning against the Reference Answer’s `<think>`.  
4. **Assign a score** (1-5) for each category based on the criteria.  
5. **Justify your score** with a **clear explanation**.  

---

### **Expected JSON Output:**
```json
{
  "Object Recognition": {
    "Score": "<1-5>",
    "Explanation": "<Brief justification>"
  },
  "Functional Reasoning": {
    "Score": "<1-5>",
    "Explanation": "<Brief justification>"
  },
  "Interaction Prediction": {
    "Score": "<1-5>",
    "Explanation": "<Brief justification>"
  }
}
```

---

**Textual Query:**  
[Insert Query]  

**Reference Answer:**  
[Insert Reference Answer]  

**LLM-generated Response:**  
[Insert LLM Response]  
"""

eval_conclusion_withoutthinktag = """You are assessing the **Conclusion** of an LLM-generated response using a structured benchmark. The benchmark consists of:

1. **A textual query** about the point cloud (serving as the evaluation prompt).
2. **A reference answer**, which contains:
   - **CoT reasoning process**: A structured reasoning process (e.g., object recognition, functional reasoning, interaction prediction).
   - **Conclusion**: A brief summary **separate from** the reasoning process.
3. **The LLM-generated response**, which follows the same structure:
   - **CoT reasoning process**: The model's reasoning process.
   - **Conclusion**: The summary that needs evaluation.

### **Your Task:**

Evaluate **only** the Conclusion in the LLM-generated response.
🚫 **Do not assess the reasoning process (CoT reasoning).**

⚠️ **Note:** Since explicit markers (e.g., `<think>`) are not present, you must distinguish the CoT reasoning process from the Conclusion based on content structure and logical flow. The Conclusion is the final summary, distinct from the step-by-step reasoning process.

-------------------------------------------------------------------------------------

#### **Evaluation Criteria:**

You will assess the **Conclusion** based on two core dimensions, each rated on a **1-5 scale** with strict scoring guidelines.

### **1. Truthfulness**

**Definition:** Does the Conclusion accurately reflect the **Reference Answer’s Conclusion**?

**Key considerations:**
 ✅ Does it correctly summarize the **Reference Answer’s Conclusion**, rather than relying solely on the reasoning process?
 ✅ If the reasoning process contains errors, does the Conclusion also inherit those errors?
 ✅ Does it introduce incorrect information that is **not present** in the Reference Answer?
 ✅ Even if the reasoning process has errors, does the Conclusion remain as correct as possible?

| Score | Evaluation Criteria                                          |
| ----- | ------------------------------------------------------------ |
| **5** | Conclusion **perfectly** matches the Reference Answer’s Conclusion—no incorrect information, even if the reasoning process has errors. |
| **4** | Conclusion **mostly** matches the Reference Answer’s Conclusion, with **minor errors** (e.g., slight wording differences, small detail omissions). |
| **3** | Conclusion **partially** matches the Reference Answer’s Conclusion but contains a **significant mistake** (e.g., incorrect object classification, function, or interaction reasoning). |
| **2** | Conclusion **largely deviates** from the Reference Answer’s Conclusion, containing **multiple errors** in reasoning or classification. |
| **1** | Conclusion is **completely incorrect**, unrecognizable in comparison to the Reference Answer’s Conclusion, or propagates all errors from the reasoning process. |

### **2. Completeness**

**Definition:** Does the Conclusion retain the **critical reasoning layers** from the CoT reasoning process (object recognition, functional reasoning, interaction prediction)?

**Key considerations:**
 ✅ Does it capture all essential layers of reasoning?
 ✅ Does it omit any important details?
 ✅ Is the information overly compressed, leading to a **loss of meaning**?

| Score | Evaluation Criteria                                          |
| ----- | ------------------------------------------------------------ |
| **5** | Conclusion **fully retains** all key information from the reasoning process (object recognition, function, interaction). No omissions. |
| **4** | Conclusion is **mostly complete**, but **slightly** lacks some details (e.g., missing one interaction type). |
| **3** | Conclusion **omits multiple key points**, such as missing function or interaction reasoning. |
| **2** | Conclusion loses **most** information, retaining only minimal useful content. |
| **1** | Conclusion **fails to extract** any key insights, offering little to no value. |

#### **Important Instruction:**

When comparing the **model’s Conclusion** to the **Reference Answer**, **only refer to the Conclusion part**. **Do not** use the reasoning process for comparison.

------

### **Expected JSON Output Format:**

Your evaluation should be returned in the following **structured JSON format**:

```
{
  "Truthfulness": {
    "Score": "<the score>",
    "Explanation": "<your explanation for this score>"
  },
  "Completeness": {
    "Score": "<the score>",
    "Explanation": "<your explanation for this score>"
  }
}
```
-------------------------------------------------------------------------------------

**Textual Query:**

[Insert the reference text here]

**Reference Answer:**

[Insert the text here]

**LLM-generated Response:**

[Insert the Response here]
"""

eval_cot_withoutthinktag = """You are an evaluator assessing the **reasoning process** of an LLM-generated response using a structured benchmark.

#### **Benchmark Components:**

1. **A textual query** about the point cloud (guiding the evaluation).
2. **A reference answer**, which consists of:
   - **CoT reasoning process**: A structured reasoning process (**step-by-step logical reasoning**).
   - **Conclusion**: A brief summary **separate from** the reasoning process (not relevant for evaluation).
3. **A generated response**, which follows the same structure:
   - **CoT reasoning process**: The model’s reasoning process (to be evaluated).
   - **Conclusion**: A summary that should be ignored.

### **Your Task:**

Evaluate **only** the reasoning process within the response.
 🚫 **Do not assess the final summary (Conclusion).**

⚠️ **Note:** Since explicit markers (e.g., `<think>`) are not present, you must distinguish the reasoning process from the Conclusion based on content structure and logical flow. The **reasoning process** consists of step-by-step logical deductions, while the **Conclusion** is a final summarized statement.

-------------------------------------------------------------------------------------

### **Evaluation Criteria**

You will assess the LLM’s reasoning based on **three categories**, each scored on a **1-5 scale** with well-defined criteria.

#### **1. Object Recognition**

**Definition:** Does the model correctly identify the object and its features based on the query?

| Score | Evaluation Criteria                                          |
| ----- | ------------------------------------------------------------ |
| **5** | The object is **correctly identified**, including precise features and relevant details. |
| **4** | The object **category is correct**, but **some specific details** are missing or slightly inaccurate. |
| **3** | The object is **partially correct**, with **some confusion** or vague description. |
| **2** | The identification is **mostly incorrect**, but some minor elements make sense. |
| **1** | The object is **entirely misidentified** or described in a way that does not match the query. |

#### **2. Functional Reasoning**

**Definition:** Does the model accurately infer the function of the object?

| Score | Evaluation Criteria                                          |
| ----- | ------------------------------------------------------------ |
| **5** | The object’s **function is accurately inferred**, with strong supporting reasoning. |
| **4** | The function is **mostly correct**, but **some details are missing** or slightly unclear. |
| **3** | The reasoning makes sense in some aspects but lacks completeness. |
| **2** | The function is **mostly incorrect**, though some logical elements exist. |
| **1** | The function is **entirely incorrect** or nonsensical.       |

#### **3. Interaction Prediction**

**Definition:** Does the model provide a **logical and coherent** description of human-object interaction?

| Score | Evaluation Criteria                                          |
| ----- | ------------------------------------------------------------ |
| **5** | The interaction is **highly logical and natural**, with precise descriptions. |
| **4** | The interaction is **mostly correct**, but lacks **some finer details**. |
| **3** | Some interactions are reasonable, but **others are unclear or missing**. |
| **2** | The interaction is **mostly incorrect** or unrealistic.      |
| **1** | The interaction is **completely wrong** or does not make sense. |

------

### **Evaluation Process**

Follow this structured process to ensure consistency:

1. **Analyze** the provided textual query and reference answer.
2. **Focus only on the reasoning process** within the LLM’s response.
3. **Compare** the reasoning process in the response against the reference answer’s reasoning process.
4. **Assign a score** (1-5) for each category based on the defined criteria.
5. **Justify your score** with a **clear explanation**.

------

### **Expected JSON Output Format**

Your evaluation should be returned in the following structured **JSON format**:

```
{
  "Object Recognition": {
    "Score": "<the score>",
    "Explanation": "<your explanation for this score>"
  },
  "Functional Reasoning": {
    "Score": "<the score>",
    "Explanation": "<your explanation for this score>"
  },
  "Interaction Prediction": {
    "Score": "<the score>",
    "Explanation": "<your explanation for this score>"
  }
}
```

-------------------------------------------------------------------------------------

**Textual Query:**

[Insert the reference text here]

**Reference Answer:**

[Insert the text here]

**LLM-generated Response:**

[Insert the Response here]
"""