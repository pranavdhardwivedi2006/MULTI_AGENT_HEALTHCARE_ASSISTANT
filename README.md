# Multi-Agent Healthcare Assistant

> An AI-based system for Symptom Analysis, Medical Report Interpretation, and Personalized Health Recommendations.

## 📖 Overview

The Multi-Agent Healthcare Assistant is an innovative AI system that distributes analysis across multiple specialized agents to collaboratively analyze patient-provided health information. Unlike traditional healthcare chatbots relying on a single conversational model, this system handles complex symptoms, multiple medical domains, patient history, and safety-critical situations through coordinated multi-agent orchestration.

**Disclaimer:** This system is designed as a decision-support and healthcare assistance tool rather than a replacement for qualified medical professionals.

## ✨ Key Features

* **Multi-Domain Symptom Analysis:** Analyzes user-provided symptoms using AI across various medical specialties.
* **Medical Report Processing:** Processes uploaded medical documents (using OCR and text extraction) to simplify and interpret complex clinical terminology.
* **Patient Memory:** Maintains contextual information such as previous conversations, report history, and health trends for personalized assistance.
* **Multi-Agent Debate & Consensus:** Enables specialized agents to compare independent analyses, identify agreements/disagreements, and generate a unified consensus.
* **Safety Verification:** Includes a dedicated safety mechanism to provide emergency guidance for serious situations and prevent excessively certain AI responses.
* **Doctor Recommendation:** Assists users in identifying potentially relevant healthcare specialists based on their symptoms and analysis.

## ⚙️ System Architecture & Workflow

The system follows a structured, modular pipeline orchestrated through a multi-agent layer:
`Patient → Frontend → Backend API → Multi-Agent Orchestrator → Planner Agent → Specialist Agents → Debate Agent → Consensus Agent → Safety Agent → Personalized Healthcare Assistance`

### Agent Roles

* **Symptom Analyzer Agent:** Performs initial analysis, extracts relevant information, and identifies possible medical domains.
* **Planner Agent:** Determines which specific specialist agents need to participate, reducing unnecessary processing.
* **Specialized Medical Agents:** Independent agents focusing on Cardiology, Neurology, Pulmonology, Gastroenterology, Endocrinology, and Dermatology.
* **Debate Agent:** Compares specialist outputs to identify agreements, missing information, supporting evidence, and uncertainty.
* **Consensus Agent:** Synthesizes the debate into a unified, structured healthcare-assistance response.
* **Safety Agent:** Reviews the final response to check for urgent medical red flags or excessive AI certainty before delivering it to the user.

## ⚠️ Limitations

* The system cannot replace clinical examinations or professional medical judgment.
* AI models may produce errors or hallucinate information.
* OCR processing may incorrectly extract text from low-quality medical documents.
* Multi-agent processing can lead to increased computational costs and response latency.
* Medical information provided by users may be incomplete.

## 👥 Contributors

* **Authors:** Maheshwar Pathak, Pranav Dhar Dwivedi, Aditya Kiran Shukla, Sachin Shood
* **Institution:** Department of Computer Science, KIET Group of Institutions
* **Project Guide:** Dr. Gaurav Dubey
