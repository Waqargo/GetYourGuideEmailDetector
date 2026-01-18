import imaplib
import email
import os
import json
import re
from bs4 import BeautifulSoup
from pymongo import MongoClient
from datetime import datetime, UTC
from dotenv import load_dotenv

# Try to import g4f, fallback to requests if not available
try:
    import g4f
    USE_G4F = True
except ImportError:
    print("[WARNING] g4f not found, using alternative free API")
    import requests
    USE_G4F = False

# ================== LOAD ENV ==================
load_dotenv()

EMAIL = os.getenv("GMAIL_ADDRESS")
APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")
MONGO_URI = os.getenv("MONGO_URI")

# ================== MONGO ==================
client = MongoClient(MONGO_URI)
db = client["Tour"]
collection = db["bookings"]

# ================== AMENDMENT DETECTION ==================
def detect_amendment(email_content, subject):
    """Detect if this is an amendment email"""
    amendment_patterns = [
        r'~~.*?~~',  # Strikethrough pattern
        r'\bNew\b.*?(?:Date|Pickup|Address|Time|Participants)',
        r'(?:Date|Pickup|Address|Time|Participants).*?\bNew\b',
        r'amended',
        r'updated',
        r'changed',
        r'modification'
    ]
    
    combined = (subject + " " + email_content).lower()
    for pattern in amendment_patterns:
        if re.search(pattern, combined, re.I):
            return True
    return False

def extract_greeting_name(email_content):
    """Extract name from greeting to filter it out"""
    greeting_patterns = [
        r'(?:Hi|Hello|Dear|Hey)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
        r'(?:Good\s+(?:morning|afternoon|evening))\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'
    ]
    
    for pattern in greeting_patterns:
        match = re.search(pattern, email_content[:500])  # Only check first 500 chars
        if match:
            return match.group(1).strip()
    return None

# ================== EMAIL CLEANING ==================
def clean_email(msg):
    """Extract email content"""
    text, html = "", ""
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            payload = part.get_payload(decode=True)
            if payload:
                try:
                    decoded = payload.decode('utf-8', errors='ignore')
                except:
                    decoded = payload.decode('latin-1', errors='ignore')
                
                if ctype == "text/plain":
                    text += decoded
                elif ctype == "text/html":
                    html += decoded
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            html = payload.decode('utf-8', errors='ignore')

    # Convert HTML to readable text
    soup = BeautifulSoup(html if html else text, "html.parser")
    clean_text = soup.get_text(separator="\n")
    
    return clean_text, html

# ================== FREE AI PARSER - METHOD 1: G4F ==================
def parse_with_g4f(email_content, subject, is_amendment=False):
    """Use GPT4Free to extract booking information"""
    
    amendment_instruction = ""
    if is_amendment:
        amendment_instruction = """
CRITICAL: This is an AMENDMENT email. Follow these rules strictly:
1. Fields marked with "New" or showing changes (strikethrough ~~old~~ new) should be extracted
2. If name or phoneNumber are NOT explicitly shown as changed/new, set them to null
3. Only extract fields that are actually present and changed in this amendment
4. Do NOT extract names from greetings like "Hi [Name]" or "Dear [Name]"
"""
    
    prompt = f"""You are parsing a GetYourGuide booking email. Extract the following information accurately.

Email Subject: {subject}

Email Content:
{email_content[:4000]}

{amendment_instruction}

Extract and return ONLY a valid JSON object with these exact fields (use null for missing data):
{{
  "booking_reference": "GYG reference number (e.g., GYG83XQWFQ7B)",
  "name": "Customer's full name (NOT email address, NOT greeting name)",
  "phoneNumber": "Customer's phone number with country code",
  "tourDate": "Tour date in format 'Month DD, YYYY' (e.g., 'March 17, 2026')",
  "tourTime": "Tour start time (e.g., '6:00 AM')",
  "totalPassengers": "Number of passengers as integer",
  "tour": "Tour/Activity name",
  "vehicleType": "Vehicle type if mentioned (e.g., 'Private Car', 'Standard Van')",
  "address": "Pickup location or meeting point address",
  "is_cancellation": true/false,
  "is_amendment": true/false
}}

CRITICAL RULES:
- "name" field must ONLY contain the person's actual name from booking details, NOT from greetings
- If name appears only in greeting context (Hi/Hello/Dear), set it to null
- "tourDate" must be the actual readable date
- All phone numbers should include country code
- Return ONLY the JSON object, no explanations
- If a field is not found or not changed (in amendments), use null
- totalPassengers must be a number, not a string"""

    try:
        response = g4f.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        response_text = str(response).strip()
        
        # Remove markdown code blocks if present
        if "```" in response_text:
            parts = response_text.split("```")
            for part in parts:
                if part.strip().startswith("json"):
                    response_text = part[4:].strip()
                    break
                elif part.strip().startswith("{"):
                    response_text = part.strip()
                    break
        
        parsed = json.loads(response_text)
        
        # Clean up booking reference
        if parsed.get("booking_reference"):
            parsed["booking_reference"] = parsed["booking_reference"].upper()
        
        # Ensure totalPassengers is an integer
        if parsed.get("totalPassengers"):
            try:
                parsed["totalPassengers"] = int(parsed["totalPassengers"])
            except:
                parsed["totalPassengers"] = None
        
        return parsed
        
    except Exception as e:
        print(f"   [WARNING] G4F Error: {str(e)[:80]}")
        return None

# ================== FREE AI PARSER - METHOD 2: Hugging Face ==================
def parse_with_huggingface(email_content, subject, is_amendment=False):
    """Use Hugging Face's free inference API"""
    
    amendment_note = ""
    if is_amendment:
        amendment_note = "This is an AMENDMENT. Only extract changed fields. Set name/phone to null if not explicitly changed."
    
    prompt = f"""Parse this GetYourGuide booking email and extract information in JSON format.

Subject: {subject}
Content: {email_content[:3000]}

{amendment_note}

Return ONLY valid JSON with these fields: booking_reference, name, phoneNumber, tourDate, tourTime, totalPassengers, tour, vehicleType, address, is_cancellation, is_amendment

Rules:
- name must be person's actual name from booking, NOT from greeting
- Use null for missing fields
- totalPassengers must be a number"""

    try:
        response = requests.post(
            "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1",
            json={
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 512,
                    "temperature": 0.1,
                    "return_full_text": False
                }
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            response_text = result[0]['generated_text'] if isinstance(result, list) else result.get('generated_text', '')
            
            # Extract JSON from response
            if "```" in response_text:
                parts = response_text.split("```")
                for part in parts:
                    if "json" in part.lower():
                        response_text = part.split("json")[1].strip()
                        break
            
            # Find JSON object
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start >= 0 and end > start:
                response_text = response_text[start:end]
            
            parsed = json.loads(response_text)
            
            # Clean up
            if parsed.get("booking_reference"):
                parsed["booking_reference"] = parsed["booking_reference"].upper()
            if parsed.get("totalPassengers"):
                parsed["totalPassengers"] = int(parsed["totalPassengers"])
            
            return parsed
        else:
            print(f"   [WARNING] HuggingFace Error: Status {response.status_code}")
            return None
            
    except Exception as e:
        print(f"   [WARNING] HuggingFace Error: {str(e)[:80]}")
        return None

# ================== COMBINED AI PARSER ==================
def parse_with_ai(email_content, subject, is_amendment=False):
    """Try multiple free AI methods until one works"""
    
    # Method 1: Try G4F first
    if USE_G4F:
        parsed = parse_with_g4f(email_content, subject, is_amendment)
        if parsed:
            return parsed
        print("   [INFO] Trying alternative method...")
    
    # Method 2: Try Hugging Face
    parsed = parse_with_huggingface(email_content, subject, is_amendment)
    if parsed:
        return parsed
    
    print("   [ERROR] All AI methods failed")
    return None

# ================== SMART MERGE FUNCTION ==================
def smart_merge(existing_booking, new_data, greeting_name):
    """Intelligently merge amendment data with existing booking"""
    
    # Protected fields - NEVER update from amendments unless explicitly changed
    protected_fields = ['name', 'phoneNumber']
    
    # Create merged document starting with existing data
    merged = existing_booking.copy()
    
    # Filter out greeting name if it was mistakenly extracted
    if new_data.get('name') == greeting_name:
        print(f"   [FILTER] Rejected greeting name: {greeting_name}")
        new_data['name'] = None
    
    # Track what was updated
    updated_fields = []
    
    # Update fields based on rules
    for field, new_value in new_data.items():
        # Skip metadata fields
        if field in ['_id', 'createdAt', 'updatedAt', 'platform', 'specialRequirements', 'is_cancellation', 'is_amendment']:
            continue
        
        # Skip if new value is None or empty
        if new_value is None or new_value == "":
            continue
        
        # For protected fields, only update if value is significantly different
        if field in protected_fields:
            # Only update if explicitly different and not a greeting
            if new_value and new_value != existing_booking.get(field):
                # Additional validation: name should not be a single word (likely greeting)
                if field == 'name' and len(new_value.split()) < 2:
                    print(f"   [FILTER] Rejected single-word name (likely greeting): {new_value}")
                    continue
                merged[field] = new_value
                updated_fields.append(field)
        else:
            # For non-protected fields, update if different
            if new_value != existing_booking.get(field):
                merged[field] = new_value
                updated_fields.append(field)
    
    # Always update timestamp
    merged['updatedAt'] = datetime.now(UTC)
    
    return merged, updated_fields

# ================== BUILD DOC ==================
def build_doc(data):
    """Build MongoDB document from parsed data"""
    now = datetime.now(UTC)
    return {
        "name": data.get("name"),
        "vehicleType": data.get("vehicleType") or "Unknown",
        "address": data.get("address"),
        "phoneNumber": data.get("phoneNumber"),
        "tour": data.get("tour"),
        "tourDate": data.get("tourDate"),
        "tourTime": data.get("tourTime"),
        "totalPassengers": data.get("totalPassengers"),
        "specialRequirements": "No",
        "booking_reference": data["booking_reference"],
        "platform": "GYG",
        "updatedAt": now
    }

# ================== MAIN ==================
def main():
    ai_method = "GPT4Free + HuggingFace" if USE_G4F else "HuggingFace"
    print(f"\n[AI BOOKING SYNC] 100% FREE ({ai_method})")
    print(f"[TIME] Started: {datetime.now().strftime('%H:%M:%S')}")
    print("="*70)

    # Connect to Gmail
    try:
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(EMAIL, APP_PASSWORD)
        mail.select("inbox")
        print("[SUCCESS] Connected to Gmail")
    except Exception as e:
        print(f"[ERROR] Gmail Login Failed: {e}")
        return

    # Search for GYG emails
    status, messages = mail.search(None, 'FROM "notification.getyourguide.com"')
    email_ids = messages[0].split()

    print(f"[INFO] Found {len(email_ids)} GetYourGuide emails")
    print(f"[INFO] Processing latest 5 emails...\n")

    processed = 0
    saved = 0
    updated = 0
    cancelled = 0
    duplicates = 0
    errors = 0

    # Process latest 5 emails
    for idx, eid in enumerate(email_ids[-5:], 1):
        try:
            print(f"\n{'='*70}")
            print(f"[EMAIL {idx}/5] ID: {eid.decode()}")
            
            res, msg_data = mail.fetch(eid, "(RFC822)")
            msg = email.message_from_bytes(msg_data[0][1])
            
            subject = msg.get("Subject", "")
            email_content, html = clean_email(msg)

            print(f"[SUBJECT] {subject[:60]}...")

            # STRICT FILTER: Only booking-related actionable emails
            if not is_allowed_booking_email(subject, email_content):
                print("[SKIPPED] Not a booking / amendment / urgent / cancellation email")
                continue

            
            # Detect if this is an amendment
            is_amendment = detect_amendment(email_content, subject)
            greeting_name = extract_greeting_name(email_content)
            
            if is_amendment:
                print(f"[DETECTED] Amendment email")
            if greeting_name:
                print(f"[DETECTED] Greeting name: {greeting_name}")
            
            print(f"[AI] Analyzing with FREE AI...")
            
            # Use AI to parse
            parsed = parse_with_ai(email_content, subject, is_amendment)
            
            if not parsed:
                errors += 1
                continue
            
            # Check if cancellation
            if parsed.get("is_cancellation"):
                booking_ref = parsed.get("booking_reference")
                if booking_ref:
                    existing = collection.find_one({"booking_reference": booking_ref})
                    if existing:
                        collection.delete_one({"booking_reference": booking_ref})
                        print(f"[CANCELLED] Deleted booking: {booking_ref}")
                        cancelled += 1
                    else:
                        print(f"[WARNING] Cancellation for {booking_ref} but not found in database")
                processed += 1
                continue
            
            # Validate booking reference
            if not parsed.get("booking_reference"):
                print(f"[SKIPPED] No booking reference found")
                continue
            
            # Check for existing booking
            booking_ref = parsed.get("booking_reference")
            existing_booking = collection.find_one({"booking_reference": booking_ref})
            
            # If this is an amendment or existing booking exists
            if existing_booking and (is_amendment or any(parsed.get(field) is None for field in ['name', 'phoneNumber'])):
                # Use smart merge
                merged_doc, updated_fields = smart_merge(existing_booking, parsed, greeting_name)
                
                # Display merged info
                print(f"\n[SMART MERGE]")
                print(f"  Reference: {merged_doc['booking_reference']}")
                print(f"  Name: {merged_doc.get('name')} [PRESERVED]")
                print(f"  Phone: {merged_doc.get('phoneNumber')} [PRESERVED]")
                
                if 'tourDate' in updated_fields:
                    print(f"  Date: {merged_doc.get('tourDate')} [UPDATED]")
                else:
                    print(f"  Date: {merged_doc.get('tourDate')}")
                
                if 'tourTime' in updated_fields:
                    print(f"  Time: {merged_doc.get('tourTime')} [UPDATED]")
                else:
                    print(f"  Time: {merged_doc.get('tourTime')}")
                
                if 'totalPassengers' in updated_fields:
                    print(f"  Passengers: {merged_doc.get('totalPassengers')} [UPDATED]")
                else:
                    print(f"  Passengers: {merged_doc.get('totalPassengers')}")
                
                if 'address' in updated_fields:
                    addr_display = str(merged_doc.get('address', ''))[:45] + "..." if len(str(merged_doc.get('address', ''))) > 45 else merged_doc.get('address', '')
                    print(f"  Address: {addr_display} [UPDATED]")
                
                if updated_fields:
                    collection.update_one(
                        {"booking_reference": booking_ref},
                        {"$set": merged_doc}
                    )
                    print(f"\n[AMENDED] Updated fields: {', '.join(updated_fields)}")
                    updated += 1
                else:
                    print(f"\n[DUPLICATE] No changes detected - skipped")
                    duplicates += 1
                    
            else:
                # New booking - build and insert
                doc = build_doc(parsed)
                
                # Display extracted info
                print(f"\n[EXTRACTED]")
                print(f"  Reference: {doc['booking_reference']}")
                print(f"  Name: {doc.get('name') or 'MISSING'}")
                print(f"  Phone: {doc.get('phoneNumber') or 'MISSING'}")
                print(f"  Date: {doc.get('tourDate') or 'MISSING'}")
                print(f"  Time: {doc.get('tourTime') or 'MISSING'}")
                print(f"  Passengers: {doc.get('totalPassengers') or 'MISSING'}")
                
                if doc.get('tour'):
                    tour_display = doc['tour'][:45] + "..." if len(doc.get('tour', '')) > 45 else doc.get('tour', '')
                    print(f"  Tour: {tour_display}")
                
                if doc.get('address'):
                    addr_display = doc['address'][:45] + "..." if len(doc.get('address', '')) > 45 else doc.get('address', '')
                    print(f"  Address: {addr_display}")
                
                collection.insert_one({
                    **doc,
                    "createdAt": datetime.now(UTC)
                })
                print(f"\n[NEW] Saved new booking")
                saved += 1
            
            processed += 1
            
        except Exception as e:
            print(f"[ERROR] {e}")
            errors += 1
            continue

    # Summary
    total = collection.count_documents({})
    print(f"\n{'='*70}")
    print(f"[SUMMARY]")
    print(f"  Processed: {processed}")
    print(f"  New Bookings: {saved}")
    print(f"  Amended: {updated}")
    print(f"  Duplicates Skipped: {duplicates}")
    print(f"  Cancelled: {cancelled}")
    print(f"  Errors: {errors}")
    print(f"  Total in Database: {total}")
    print(f"{'='*70}\n")
    
    mail.logout()
    print("[COMPLETE] Sync finished\n")
# ================== STRICT BOOKING TYPE FILTER ==================
def is_allowed_booking_email(subject, email_content):
    """
    Allow ONLY:
    - New booking
    - Booking detail change / amendment
    - Urgent booking
    - Urgent: New booking received
    - Cancel booking
    """

    allowed_patterns = [
        r'new booking',
        r'booking',
        r'booking confirmation',
        r'booking details',
        r'booking detail change',
        r'change booking',
        r'amend(?:ment|ed)?',
        r'update(?:d)? booking',
        r'urgent booking',
        r'Urgent: New booking received',
        r'cancel(?:lation|led)? booking',
        r'booking cancelled'
    ]

    combined = f"{subject} {email_content}".lower()

    return any(re.search(pattern, combined, re.I) for pattern in allowed_patterns)

if __name__ == "__main__":
    main()