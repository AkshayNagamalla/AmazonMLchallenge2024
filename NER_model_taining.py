import pandas as pd
import re
import spacy
from spacy.tokens import DocBin
from tqdm import tqdm
import random
from spacy.util import filter_spans
import subprocess
from multiprocessing import Pool


def run_command(enitity):
    """
    Train NER model
    """
    command = "python -m spacy train config.cfg --output ./output__"+enitity+" --paths.train ./train__"+enitity+".spacy --paths.dev ./dev__"+enitity+".spacy"
    subprocess.run(command, shell=True)



def process(enitity):
    """
    pipeline for Named Entity Recognistion (preprocessing and training)
    NOTE : Text from Images are extracted and stored in `ocr_result.csv` using PaddleOCR
    """
    df = pd.read_csv("../dataset/ocr_results.csv")
    df_train = pd.read_csv("../dataset/train.csv")
    
    df['Image Path'] = df['Image Path'].apply(lambda x: x.split("\\")[-1])
    df_train["image_link"] = df_train['image_link'].apply(lambda x:x.split('/')[-1])
    
    df_train.rename(columns={'image_link': 'Image'}, inplace=True)
    df.rename(columns={'Image Path': 'Image'}, inplace=True)
    
    data = pd.merge(df_train, df, on = "Image")
    
    d = data[data['entity_name']==enitity]

    def preprocess_text(text):

        """
        Replace unit misspellings and handle units attached to numbers
        """
        unit_replacements = {
            # Length and dimension units
            r'(\d+(\.\d+)?)\s*\'' : r' \1 foot ',  # Single quote for feet
            r'(\d+(\.\d+)?)\s*\"' : r' \1 inch ',  # Double quote for inches
            r'(\d+(\.\d+)?)\s*(in|In|"|\'|inch|Inch|inchs|inches|Inches)\b': r' \1 inch ',
            r'(\d+(\.\d+)?)\s*(ft|FT|feet|Feet|foot|Foot)\b': r'\1 foot ',
            r'(\d+(\.\d+)?)\s*(cm|CM|centimeters|Centimeters|centimetre|Centimetre)\b': r' \1 centimetre ',
            r'(\d+(\.\d+)?)\s*(m|M|metre|Metre|meters|Meters)\b': r' \1 metre ',
            r'(\d+(\.\d+)?)\s*(mm|MM|millimeters|Millimeters|millimetre|Millimetre)\b': r' \1 millimetre ',
            r'(\d+(\.\d+)?)\s*(yard|Yard|yards|Yards)\b': r'\1 yard ',
            
            # Weight units
            r'(\d+(\.\d+)?)\s*(g|G|grams|Grams|gram|Gram)\b': r' \1 gram ',
            r'(\d+(\.\d+)?)\s*(kg|KG|kilograms|Kilograms|kilogram|Kilogram)\b': r' \1 kilogram ',
            r'(\d+(\.\d+)?)\s*(mg|MG|milligrams|Milligrams|milligram|Milligram)\b': r' \1 milligram ',
            r'(\d+(\.\d+)?)\s*(lb|1b|1bs|LB|lbs|LBS|pounds|Pounds|pound|Pound)\b': r' \1 pound ',
            r'(\d+(\.\d+)?)\s*(oz|OZ|0z|ounces|Ounces|ounce|Ounce)\b': r' \1 ounce ',
            r'(\d+(\.\d+)?)\s*(ton|Ton|tons|Tons)\b': r' \1 ton ',

            # Volume units
            r'(\d+(\.\d+)?)\s*(l|L|liters|Liters|litres|Litres|litre|Litre)\b': r' \1 litre ',
            r'(\d+(\.\d+)?)\s*(ml|ML|milliliters|Milliliters|millilitres|Millilitres|millilitre|Millilitre)\b': r' \1 millilitre ',
            r'(\d+(\.\d+)?)\s*(cl|CL|centiliters|Centiliters|centilitre|Centilitre)\b': r' \1 centilitre ',
            r'(\d+(\.\d+)?)\s*(dl|DL|deciliters|Deciliters|decilitre|Decilitre)\b': r' \1 decilitre ',
            r'(\d+(\.\d+)?)\s*(microlitre|Microlitre|microliters|Microliters|µL|uL)\b': r' \1 microlitre ',
            r'(\d+(\.\d+)?)\s*(pint|Pint|pints|Pints)\b': r' \1 pint ',
            r'(\d+(\.\d+)?)\s*(quart|Quart|quarts|Quarts)\b': r' \1 quart ',
            r'(\d+(\.\d+)?)\s*(cup|Cup|cups|Cups)\b': r' \1 cup ',
            r'(\d+(\.\d+)?)\s*(gallon|Gallon|gallons|Gallons)\b': r' \1 gallon ',
            r'(\d+(\.\d+)?)\s*(imperial gallon|Imperial Gallon|imperial gallons|Imperial Gallons)\b': r' \1 imperial gallon ',
            r'(\d+(\.\d+)?)\s*(cubic inch|Cubic Inch|cubic inches|Cubic Inches)\b': r' \1 cubic inch ',
            r'(\d+(\.\d+)?)\s*(cubic foot|Cubic Foot|cubic feet|Cubic Feet)\b': r' \1 cubic foot ',
            r'(\d+(\.\d+)?)\s*(fl oz|FL OZ|fluid ounce|Fluid Ounce|fluid ounces|Fluid Ounces)\b': r' \1 fluid ounce ',

            # Voltage units
            r'(\d+(\.\d+)?)\s*(volt|Volt|volts|Volts|v|V)\b': r' \1 volt ',
            r'(\d+(\.\d+)?)\s*(kilovolt|Kilovolt|kV|KV)\b': r' \1 kilovolt ',
            r'(\d+(\.\d+)?)\s*(millivolt|Millivolt|mV|MV)\b': r' \1 millivolt ',

            # Power units
            r'(\d+(\.\d+)?)\s*(watt|Watt|watts|Watts|w|W)\b': r' \1 watt ',
            r'(\d+(\.\d+)?)\s*(kilowatt|Kilowatt|kW|KW)\b': r' \1 kilowatt '
        }
        
        # Replace unit misspellings and handle units attached to numbers
        for pattern, replacement in unit_replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text


    def find_matching_indices(text, entity_value):
        """
        find start and stop index of Entity in String (used for NER model Training)
        """
        # Split the entity value into numerical and unit parts
        parts = entity_value.split()
        if len(parts) != 2:
            return (-1, -1)  # Invalid format
    
        num_part = parts[0]
        unit_part = parts[1]
    
        # Remove trailing .0 from the numerical part if it is an integer
        normalized_num_part = num_part if not num_part.endswith('.0') else num_part[:-2]
        
        # Define regex patterns
        primary_pattern = rf'\b{re.escape(normalized_num_part)}\s+{re.escape(unit_part)}\b'
        alternative_pattern = rf'\b{re.escape(num_part)}\s+{re.escape(unit_part)}\b'
        
        # Search for the primary pattern
        match = re.search(primary_pattern, text, re.IGNORECASE)
        
        if match:
            # Return indices for primary pattern
            start_index = match.start()  # 0-based index
            end_index = match.end()   # 0-based index, inclusive end index
            return (start_index, end_index)
        
        # If primary pattern not found, search for the alternative pattern
        if normalized_num_part != num_part:
            match = re.search(alternative_pattern, text, re.IGNORECASE)
            
            if match:
                # Return indices for alternative pattern
                start_index = match.start()  # 0-based index
                end_index = match.end() - 1  # 0-based index, inclusive end index
                return (start_index, end_index)
        
        # Return -1 if neither pattern is found
        return (-1,-1)

    d.loc[:, "Extracted Text"] = d["Extracted Text"].apply(preprocess_text)
    
    d_copy = d.copy()
    
    d_copy[['Start Index', 'End Index']] = d_copy.apply(
        lambda row: pd.Series(find_matching_indices(row['Extracted Text'], row['entity_value'])),
        axis=1
    )
    d_copy = d_copy[(d_copy['Start Index'] != -1) & (d_copy['End Index'] != -1)]

    
    
    
    nlp = spacy.blank("en")
    doc_bin = DocBin()
    
    def df_to_list_of_dict(d_copy):
        training_data = []
        
        # Use .iterrows() to iterate over DataFrame rows
        for index, row in d_copy.iterrows():
            temp = {}
            temp["text"] = row["Extracted Text"]
        
            # Initialize the 'entity' dictionary
            temp["entity"] = {
                "start": row["Start Index"],
                "end": row["End Index"],
                "label": row["entity_name"]
            }
        
            training_data.append(temp)
        return training_data
    
    
    sc = 0  # Counter for skipped spans
    
    d__ = d_copy.sample(frac = 0.70, random_state=42)
    train = df_to_list_of_dict(d__)
    dev = df_to_list_of_dict(d_copy.drop(d__.index))
    
    # Create separate DocBin objects for train and dev data
    doc_bin_train = DocBin()
    doc_bin_dev = DocBin()
    
    # Process and store training data
    for data in tqdm(train):
        text = data["text"]
        labels = data["entity"]
        doc = nlp.make_doc(text)
    
        start = labels["start"]
        end = labels["end"]
        label = labels["label"]
        
        # Create a span for the entity
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
    
        ents_ = []
        if span is None:
            print(f"Skipping example with text in {enitity}: {text}")
            sc += 1
        else:
            ents_.append(span)
    
        doc.ents = ents_
        doc_bin_train.add(doc)  # Add the document to the train DocBin
    
    # Save the training data to a .spacy file
    doc_bin_train.to_disk("train__"+enitity+".spacy")
    
    
    
    
    
    # Process and store development data
    for data in tqdm(dev, desc="Processing dev data"):
        text = data["text"]
        labels = data["entity"]
        doc = nlp.make_doc(text)
    
        start = labels["start"]
        end = labels["end"]
        label = labels["label"]
        
        # Create a span for the entity
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
    
        ents_ = []
        if span is None:
            print(f"Skipping example with text in {enitity}: {text}")
            sc += 1
        else:
            ents_.append(span)

        doc.ents = ents_
        doc_bin_dev.add(doc)

    # Save the development data to a .spacy file
    doc_bin_dev.to_disk("dev__"+enitity+".spacy")
    
    print(f"Total skipped examples {enitity}: {sc}")

    run_command(enitity)

    return 1 


if __name__ == "__main__":
    e = ['item_weight', 'item_volume', 'voltage', 'wattage','maximum_weight_recommendation', 'height', 'depth', 'width']

    for i in e:
        process(e)

    print("8 Models Trained Sucessfully")