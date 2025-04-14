# main.py

# Import necessary libraries and modules
import os
import json
import re
import pdfplumber
import pandas as pd 
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from crewai import Agent, Task, Crew, Process
from pydantic import BaseModel, ValidationError
from typing import List, Dict, Any

# Load environment variables from the .env file
load_dotenv()
os.environ["OPENAI_API_KEY"]  # Set OpenAI API key from environment variables

# Define the model name for OpenAI
gpt_model = "gpt-4o-mini"

# Initialize Flask app
app = Flask(__name__)

# Define a function to extract text from a PDF file using pdfplumber
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:  # Open PDF file
        text = []
        for page in pdf.pages:  # Iterate through each page
            text.append(page.extract_text(x_tolerance=1, y_tolerance=1))  # Extract text with tolerances
        return "\n".join(text)  # Join all text as a single string
    
# Function to extract 200 characters of context after any matching total amount keyword
def extract_total_value_context(text, keywords=None, chars_after=200):
    # Default keywords to look for if none are provided
    if keywords is None:
        keywords = [
            "Total Order Value",
            "Grand Total",
            "Total(INR)",
            "TOTAL AMOUNT",
            "Total Amount",
            "Total Payable",
            "Invoice Total",
            "Amount Payable",
            "Net Payable",
            "TOTAL VALUE"
        ]
    
    # Iterate through each keyword
    for keyword in keywords:
        # Search for the keyword followed by any text on the same line
        match = re.search(rf"{re.escape(keyword)}[^\n]*", text, re.IGNORECASE)
        if match:
            start_index = match.end()
            return text[start_index:start_index + chars_after].strip()
    
    # If no keyword matches, return an empty string
    return ""

# Define a function to process and consolidate line items from the extracted text
def consolidate_line_items(po_text):
    lines = po_text.split('\n')  # Split by new line
    merged_lines = []  # Final processed lines
    buffer = ''  # Temporary storage for multi-line merge
    for line in lines:
        line = line.strip()  # Remove whitespace
        # Skip lines with irrelevant price labels
        if re.search(r'(MRP|List\s*Price|Maximum\s*Retail\s*Price)', line, re.IGNORECASE):
            continue
        # Push buffer when meaningful line appears
        if re.search(r'(Sale\s*Price|Selling\s*Price|Unit\s*Price|Landed\s*Price)', line, re.IGNORECASE):
            if buffer:
                merged_lines.append(buffer)
                buffer = ''
            merged_lines.append(line)
            continue
        # If line has EAN, start new buffer
        if re.search(r'\b\d{11,13}\b', line):
            if buffer:
                merged_lines.append(buffer)
            buffer = line
        else:
            buffer += ' ' + line  # Continue appending to buffer
    if buffer:
        merged_lines.append(buffer)  # Final buffer push
    return '\n'.join(merged_lines)

# Define data models using Pydantic for type validation
class LineItem(BaseModel):
    product_id: str
    hsn_code: str
    ean: str
    product_name: str
    qty: int
    unit_price_incl_tax: float
    line_total: float
    price_breakdown: Dict[str, float] = {}  # Optional extra pricing fields

class PurchaseOrder(BaseModel):
    external_order_number: str
    customer_name: str
    customer_gst: str
    ship_to_location_address: str
    ship_to_city: str
    ship_to_pincode: str
    total_line_items: int
    total_item_qty_in_units: int
    order_total_amount_incl_tax: float
    line_item_details: List[LineItem]  # List of validated line items

class ValidatedOutput(BaseModel):
    data: Dict[str, Any]  # Validated data structure
    errors: List[str]  # All validation errors
    accuracy: float  # Accuracy score
    iteration: int  # Iteration count

# Updated Hard Validation Logic for Line Level Validation
def hard_validate(data: Dict[str, Any]) -> ValidatedOutput:
    errors = []  # List to collect any error messages
    validated_data = data.copy()  # Make a copy to avoid modifying original input

    faulty_indices_initial = set()  # Track line items with initial price/qty/line_total errors before validation
    faulty_indices_step1 = set()    # Track line items still wrong after line-item level validation (Step 1)


    # Total number of line items
    total_items = len(validated_data.get('line_item_details', []))

    # ACCURACY 1: Before validation (raw errors from parser) 
    # Check which line items are already clearly wrong (e.g. qty, unit_price, line_total = 0)
    for idx, item in enumerate(validated_data.get('line_item_details', [])):
        if item.get("qty", 0) == 0 or item.get("line_total", 0) == 0 or item.get("unit_price_incl_tax", 0) == 0:
            faulty_indices_initial.add(idx)

    # Calculate accuracy_1 before validation
    if total_items == 0:
        accuracy_1 = 0.0
    else:
        accuracy_1 = ((total_items - len(faulty_indices_initial)) / total_items) * 100  # Pre-validation accuracy

    # STEP 1: Line-item level validation 
    for idx, item in enumerate(validated_data.get('line_item_details', [])):
        qty = item.get('qty', 0)  # Extract quantity
        unit_price = item.get('unit_price_incl_tax', 0)  # Extract unit price
        line_total = item.get('line_total', 0)  # Extract line total
        breakdown_values = list(item.get('price_breakdown', {}).values())  # Extract price breakdown fields

        # If quantity is zero, it's an invalid item
        if qty == 0:
            errors.append(f"Line item {idx+1}: Quantity can not be 0")
            faulty_indices_step1.add(idx)  # Mark this index as faulty after step 1
            continue

        # Direct check: Is line_total = qty * unit_price?
        if -1 < ((qty * unit_price) - line_total) < 1:
            continue  # Valid match, skip correction

        # GST logic: Check if GST-adjusted unit price gives valid total
        unit_price_gst = round(unit_price * 1.18, 2)
        if -1 < ((qty * unit_price_gst) - line_total) < 1:
            item['unit_price_incl_tax'] = unit_price_gst  # Accept GST-corrected value
            continue

        # Try correcting using permutation of price breakdown values
        all_values = [unit_price, line_total] + breakdown_values  # Combine all price-related fields
        all_values = sorted(set(all_values), reverse=True)  # Deduplicate and sort descending

        # Guess new line total from top price value
        if qty > 1:
            new_line_total = all_values[0]
        else:
            new_line_total = all_values[1] if len(all_values) > 1 else all_values[0]

        # Check if guessed line total works with original unit price
        if -1 < ((qty * unit_price) - new_line_total) < 1:
            item['line_total'] = new_line_total
            item['unit_price_incl_tax'] = unit_price
            continue

        # Try GST-adjusted unit price with guessed total
        if -1 < ((qty * unit_price_gst) - new_line_total) < 1:
            item['line_total'] = new_line_total
            item['unit_price_incl_tax'] = unit_price_gst
            continue

        # Validation failed, but we won’t count EAN-only issues here
        faulty_indices_step1.add(idx)
        
        # Still not valid — log the issue
        errors.append(
            f"""
            These are the errors:
            Line item {idx+1}: Validation failed for `unit_price_incl_tax` and `line_total` even after applying correction logic.
            Original Extracted Values from Parser:
              - unit_price_incl_tax (parser): {data['line_item_details'][idx].get('unit_price_incl_tax')}
              - qty (parser): {data['line_item_details'][idx].get('qty')}
              - line_total (parser): {data['line_item_details'][idx].get('line_total')}
            Suggestion: Let the Corrector Agent re-evaluate this using semantic cues and domain rules from Indian POs.
            """
        )

        # EAN validation logic (excluded from accuracy calculation)
        raw_ean = item.get('ean', '').strip()
        split_parts = re.findall(r'\d{11,13}', raw_ean)
        item['ean'] = split_parts[0] if split_parts else raw_ean

        if not re.match(r'^\d{11,13}$', item['ean']):
            errors.append(f"Line item {idx+1}: Invalid EAN after cleanup → {item['ean']}")

        # HSN Code validation
        if not re.match(r'^\d{4,8}$', item.get('hsn_code', '')):
            errors.append(f"Line item {idx+1}: Invalid HSN Code")
            faulty_indices_step1.add(idx)

    # ACCURACY 2: After Step 1 line-item validation 
    if total_items == 0:
        accuracy_step1 = 0.0
    else:
        accuracy_step1 = ((total_items - len(faulty_indices_step1)) / total_items) * 100

    # STEP 2: PO-level total validation 
    order_total = data.get('order_total_amount_incl_tax', 0)
    line_items = validated_data.get('line_item_details', [])
    total_sum = sum(item.get('line_total', 0) for item in line_items)

    if -1 < (order_total - total_sum) < 1:
        for item in line_items:
            item['line_total'] = round(item['line_total'], 2)  # Ensure rounding consistency

        for idx, item in enumerate(line_items):
            qty = item.get('qty', 0)
            if qty == 0:
                continue
            unit_price = round(item['line_total'] / qty, 2)

            if -1 < ((unit_price * qty) - item['line_total']) < 1:
                item['unit_price_incl_tax'] = unit_price
                if idx in faulty_indices_step1:
                    faulty_indices_step1.remove(idx)  # If fixed, remove from faulty
                continue

            unit_price_gst = round(unit_price * 1.18, 2)
            if -1 < ((unit_price_gst * qty) - item['line_total']) < 1:
                item['unit_price_incl_tax'] = unit_price_gst
                if idx in faulty_indices_step1:
                    faulty_indices_step1.remove(idx)
                continue

    elif -1 < (order_total - (total_sum * 1.18)) < 1:
        for item in line_items:
            item['line_total'] = round(item['line_total'] * 1.18, 2)

        for idx, item in enumerate(line_items):
            qty = item.get('qty', 0)
            if qty == 0:
                continue
            unit_price = round(item['line_total'] / qty, 2)

            if -1 < ((unit_price * qty) - item['line_total']) < 1:
                item['unit_price_incl_tax'] = unit_price
                if idx in faulty_indices_step1:
                    faulty_indices_step1.remove(idx)
                continue

            unit_price_gst = round(unit_price * 1.18, 2)
            if -1 < ((unit_price_gst * qty) - item['line_total']) < 1:
                item['unit_price_incl_tax'] = unit_price_gst
                if idx in faulty_indices_step1:
                    faulty_indices_step1.remove(idx)
                continue

    else:
        errors.append("Order Total Validation Error: Neither direct nor GST-adjusted line totals match order_total_amount_incl_tax")

    # ACCURACY 3: After Step 2 (PO-level total correction) 
    if total_items == 0:
        accuracy_step2 = 0.0
    else:
        accuracy_step2 = ((total_items - len(faulty_indices_step1)) / total_items) * 100

    # Final accuracy is same as accuracy after Step 2
    final_accuracy = accuracy_step2

    # Return all 3 accuracies along with the result
    return ValidatedOutput(
        data={
            **validated_data,
            "accuracy_1": round(accuracy_1, 2),
            "accuracy_step1": round(accuracy_step1, 2),
            "accuracy_step2": round(accuracy_step2, 2)
        },
        errors=errors,
        accuracy=round(final_accuracy, 2),
        iteration=0
    )

# Factory function for creating agents
def create_agents():
    llm = ChatOpenAI(model=gpt_model, temperature=0)

    parser_agent = Agent(
        role="PO Parsing Expert",
        goal="Extract all price columns and map strictly",
        backstory="Strict FMCG PO parser",
        llm=llm, verbose=True, memory=True
    )

    # soft_validator_agent = Agent(
    #     role="Semantic Validator",
    #     goal="Catch logical and tax inconsistencies",
    #     backstory="Compliance analyst",
    #     llm=llm, verbose=True
    # )

    # corrector_agent = Agent(
    #     role="PO Data Corrector",
    #     goal="Fix missing/incorrect fields based on Indian PO norms",
    #     backstory="Expert in PO correction",
    #     llm=llm, verbose=True
    # )

    return parser_agent #, soft_validator_agent, corrector_agent

# JSON cleaner to sanitize output from LLM
def clean_json(raw_text):
    if not raw_text or raw_text.strip() == "":
        return None
    cleaned = re.sub(r"```json|```", "", raw_text).strip()
    cleaned = re.sub(r'//.*', '', cleaned)
    cleaned = re.sub(r'(\d)_(\d)', r'\1\2', cleaned)
    cleaned = cleaned.replace("None", "null").replace("True", "true").replace("False", "false")
    cleaned = re.sub(r',\s*}', '}', cleaned)
    cleaned = re.sub(r',\s*]', ']', cleaned)
    cleaned = re.sub(r'\"([^\"]*?)\'(.*?)\'([^\"]*?)\"', r'"\1\2\3"', cleaned)
    cleaned = re.sub(r'(\d)\n\s*\"', r'\1,\n\"', cleaned)
    cleaned = re.sub(r'(\d),(\d{3})(\.\d+)?', r'\1\2\3', cleaned)
    return cleaned

import requests

def get_confirm_token(response):
    # Google sometimes requires a confirmation token for large or flagged files
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def download_pdf_from_google_drive(file_id, save_path="temp_po.pdf"):
    """
    Downloads a PDF from Google Drive using file_id and saves it locally.
    """
    URL = "https://docs.google.com/uc?export=download"

    # Start a session to persist cookies
    session = requests.Session()

    # Initial request to get the download page
    response = session.get(URL, params={'id': file_id}, stream=True)

    # Try to extract confirmation token if present
    token = get_confirm_token(response)

    # If a token was found, request the file with confirmation
    if token:
        response = session.get(URL, params={'id': file_id, 'confirm': token}, stream=True)

    # Check if the final response is successful
    if response.status_code == 200:
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(32768):  # Download in chunks
                if chunk:
                    f.write(chunk)
        print(f"[INFO] File downloaded and saved to: {save_path}")
        return save_path
    else:
        raise Exception("Failed to download PDF. Check if file is public or if ID is correct.")

# API endpoint
# Endpoint to process extracted text directly via POST request
@app.route('/process', methods=['POST'])
def process_po_from_gdrive():
    try:
        # Get JSON data from POST request
        request_data = request.get_json()

        # Extract 'file_id' key from the JSON payload
        file_id = request_data.get('file_id')

        # Validate presence of file_id
        if not file_id:
            return jsonify({"error": "Missing 'file_id' in request"}), 400

        pdf_path = download_pdf_from_google_drive(file_id)

        # Extract raw text from the downloaded PDF
        raw_text = extract_text_from_pdf(pdf_path)

        # Extract amount-related context
        total_context = extract_total_value_context(raw_text)
        print("[DEBUG] Total Amount Context:")
        print(total_context)

        # Merge and clean PO line item section
        processed_text = consolidate_line_items(raw_text)

        # Create the parser agent
        parser_agent = create_agents()

        print("Running parse task")

        # Convert Pydantic model schema to JSON schema string for prompt injection
        purchase_order_dict = PurchaseOrder.model_json_schema()
        purchase_order_json = json.dumps(purchase_order_dict, indent=2).replace('{', '{{').replace('}', '}}')

        # Define parsing task
        parse_task = Task(
        description=f"""Extract and normalize Indian Purchase Order data.

        ### MANDATORY RULES:
        1. **Primary ID (product_id) Hierarchy**: 
            - 'Use "Material Code" if available (e.g., 4000000xxxx)' or 'use "Article No" if Available (e.g., 14000xxxx)' for "product_id" or 'use "SKU code" if available without taking any space for "product_id"'.
            - Else fallback to **EAN** (13 digit numeric) for "product_id". If `ean` has space-separated parts like `88061825571 25`, extract only the **13-digit valid code** like '8806182557125'.
        2. **HSN Code**: Mandatory 4 to 8-digit HSN code (e.g., 33049910, 34013090). Take without any space like '33041000' if '33041 000'.
        3. **Product Description**: Merge multi-line descriptions into one line (handle wrapped text in PDFs).
        4. **Buyer Information**: Extract GSTIN (Don't extract this "GSTIN: 27AAKCM9228M1ZR", other than this string "27AAKCM9228M1ZR" you can pick for GSTIN ), PAN, Company Name, Billing & Shipping Address.
        5. **Total Amounts** or **order_total_amount_incl_tax**: Validate against "Total(INR)" or "Grand Total(INR)" in the PO. Total value shall be inclusive GST and it is most highest amount or Value in the PO data.
        6. **Pincode**: Always extract a valid 6-digit pincode (e.g., 400018, 421302).
        7. **Shipping & Billing Address**: Separate fields for both from the document layout.
        8. **Prepared/Verified By**: Extract these if mentioned.
        9. **Price Reasonability**:
            - Cosmetics: ₹50 to ₹4000.
        10. For each line item:
            - Extract and label only the "Unit Price (landed Price)" or "Selling Price" under the key "unit_price_incl_tax".
            - Extract `qty`, qty is a numeric value. ensure you pick right quantity incoherence with item total. If you have "total_line_items"== 1, then 'qty' must not be 1 unless 1 is explicitly written as the quantity (e.g., near a product name or EAN).
            - 'ean', 'hsn_code', and 'description' must be extracted accurately and distinctly, ensuring no overlap or mixing of values.
            - DO NOT extract or use MRP, base price, or other irrelevant prices in calculations strictly with followed by condition.
            - Validate EAN, and HSN formats per Indian norms.
        11. This is must. I need atleast 4 values here. For each line item in the input text extract values which look '^\d{{1,4}}.\d{{2}}$' or '^\d{{1,4}}$' pick them and store them to line_item_details array. for each line and tag each value as item1, item2, item3, item4, item5... atlest extract 3 values. pick all possible value, if there is no value fill with zero.
        12. **total_item_qty_in_units** will be always "sum of 'qty' of each line item".
        13. Number of "total_line_items" will always depends on uniqueness of Primary id (product_id) and product_name (Product Description)

        ### BUSINESS CONTEXT:
            - Indian PO conventions from sectors like FMCG, beauty products, and retail.
            - Address hierarchy: Street, City, District, State, Pincode.
            - GST: Validate GST codes based on the state (e.g., 27 = Maharashtra, 06 = Haryana, 29 = Karnataka), Don't pick this GSTIN: '27AAKCM9228M1ZR', other than this string you can pick for GSTIN.

        ### OUTPUT:
        Strict JSON conforming to PurchaseOrder schema, with extra metadata if available. Here is the json schema f{purchase_order_json}.

        ### INPUT TEXT:
        {processed_text}
        """,
        agent=parser_agent,
        expected_output="Strict JSON output matching PurchaseOrder schema with additional metadata where applicable"
    )

        # Run CrewAI parsing task
        crew = Crew(agents=[parser_agent], tasks=[parse_task], process=Process.sequential)
        parse_output = crew.kickoff()

        # Clean LLM response and load it as dictionary
        parsed_data = json.loads(clean_json(parse_output.raw))

        # Save parser output to local file (optional)
        os.makedirs("results", exist_ok=True)
        with open("results/parsed_data.txt", "w") as file:
            json.dump(parsed_data, file, indent=4)

        # Validation iteration logic
        iteration, max_iterations, final_output = 0, 1, None

        while iteration < max_iterations:
            validation = hard_validate(parsed_data)
            print("\n[DEBUG] Validation Result (After Parser):")
            with open("results/validation_data.txt", "w") as file:
                json.dump(validation.data, file, indent=4)
            print(json.dumps(validation.data, indent=2))

            # Load the mapping CSV for customer-location mapping
            mapping_df = pd.read_csv("Cid(GStName) Lid(GstPin).csv", on_bad_lines='skip')  
           
            #Normalize the column names (remove leading/trailing whitespace)
            mapping_df.columns = mapping_df.columns.str.strip()
            
            # Clean and standardize the relevant fields in the DataFrame
            mapping_df['locationName'] = mapping_df['locationName'].astype(str).str.strip().str.lower()  # Normalize location name for case-insensitive match
            mapping_df['GSTIN'] = mapping_df['GSTIN'].astype(str).str.strip()  # Clean GSTIN column
            mapping_df['pinCode'] = mapping_df['pinCode'].astype(str).str.replace(".0", "", regex=False).str.strip()  # Convert float-like strings to pure 6-digit pin

            # Extract values from the parsed PO data (already validated)
            customer_name_norm = parsed_data.get("customer_name", "").strip().lower()  # Normalize customer name from LLM output
            customer_gst = parsed_data.get("customer_gst", "").strip()  # GST from parser output
            ship_to_pincode = str(parsed_data.get("ship_to_pincode", "")).strip()  # Convert pinCode to string and trim whitespace

            # Attempt to find matching row for customer ID based on location name and GSTIN
            matched_customer = mapping_df[
                (mapping_df["locationName"] == customer_name_norm) &  # match on location name
                (mapping_df["GSTIN"] == customer_gst)  # match on GSTIN
            ]
            # Get customer ID if match found, else set "NOT FOUND"
            customer_id = str(matched_customer.iloc[0]["customerID"]) if not matched_customer.empty else "NOT FOUND"

            # Attempt to find matching row for location ID using pinCode and GSTIN
            matched_location = mapping_df[
                (mapping_df["pinCode"] == ship_to_pincode) &  # match on pin code
                (mapping_df["GSTIN"] == customer_gst)  # match on GSTIN
            ]
            # Get location ID if match found, else set "NOT FOUND"
            location_id = str(matched_location.iloc[0]["locationID"]) if not matched_location.empty else "NOT FOUND"

            # Add both values to the final output dictionary before returning it
            validation.data["customer_id"] = customer_id
            validation.data["location_id"] = location_id

            final_output = validation
            break

        # Pydantic schema validation
        try:
            PurchaseOrder(**final_output.data)
            final_output.errors.append("Final schema validation passed")
        except (ValidationError, json.JSONDecodeError, ValueError) as e:
            final_output.errors.append(f"Schema validation failed: {str(e)}")

        # Return structured validated output to Postman
        return jsonify(final_output.data), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)