import pandas as pd
import os
import re
from src.valid_index import ValidIndex



def process_regulations(filenames_as_list, valid_index_checker, non_text_labels):
    all_data_as_lines = []
    non_text = {}
    for i in range(0, len(non_text_labels)):
        non_text[non_text_labels[i]] = {}

    for file in filenames_as_list:
        #print(f"Processing file: {file}")
        text = {}
        with open(file, 'r', encoding='utf-8') as f:
            text = f.read()

        remaining_lines = text.split('\n')
        # remove empty lines
        remaining_lines = [line for line in remaining_lines if line.strip() != '']

        for i in range(0, len(non_text_labels)):
            remaining_lines, non_text_table_dict = extract_non_text(remaining_lines, '#' + non_text_labels[i])
            #print(non_text_table_dict)
            non_text[non_text_labels[i]].update(non_text_table_dict)

        # print(f'read {len(remaining_lines)} lines from {file})')
        all_data_as_lines.extend(remaining_lines)
        #print(f"Added: {len(remaining_lines)} lines")

    df = pd.DataFrame()
    df = process_lines(all_data_as_lines, valid_index_checker)
    add_full_reference(df, valid_index_checker) # adds the reference to the input dataframe
    df['word_count'] = df['Text'].str.split().str.len()
    return df, non_text



def process_lines(lines, valid_index_checker):
    data = {
        "Indent": [],
        "Reference": [],
        "Text": [],
        "Document": [],
        "Page": [],
        "Heading": []
    }

    for line_of_text in lines:
        if line_of_text.strip() != '':  # Skip blank lines
            # Find and remove any special markup characters from the line of text
            pattern = r'\(([^\(\)]*\.pdf); pg (\d+)\)\s*$'
            # Use search to find matches
            document_page = re.search(pattern, line_of_text)
            if document_page:
                data['Document'].append(document_page.group(1).strip())
                data['Page'].append(document_page.group(2).strip())
                line_of_text = line_of_text[:document_page.start()] + line_of_text[document_page.end():]
            else:
                data['Document'].append('')
                data['Page'].append('')

            pattern = '\(#Heading\)'
            # Use search to find matches
            document_page = re.search(pattern, line_of_text)
            if document_page:
                data['Heading'].append(True)
                line_of_text = line_of_text[:document_page.start()] + line_of_text[document_page.end():]
            else:
                data['Heading'].append(False)

            #Now strip out the index part
            indent, reference, remaining_text = valid_index_checker.parse_line_of_text(line_of_text)

            data['Indent'].append(indent)
            data['Reference'].append(reference.strip())
            data['Text'].append(remaining_text.strip())

    df = pd.DataFrame(data)
    return df


# TODO: Remove the page reference from the 'block_identifier' because this is messing up the dictionary keys
def extract_non_text(lines, block_identifier, hard_stop = 100):
    """
    Very crude function to extract the following blocks from the text:
            - block_identifier = 'Table', 'Formula' or 'Example'
    from the text
    """
    dictionary = {}
    current_block = None
    line_counter = 0
    remaining_lines = []


    for line in lines:
        stripped_line = line.lstrip(' ')
        if stripped_line.startswith(block_identifier):
            if '- end' in line:
                current_block = None
                line_counter = 0
            else:
                current_block = stripped_line.strip()
                dictionary[current_block] = []
            continue

        if current_block is not None:
            if line_counter < hard_stop:
                #dictionary[current_block].append(stripped_line)
                dictionary[current_block].append(line)
                line_counter += 1
            else:
                raise ValueError(f'Formatting issue with {current_block}: more than {hard_stop} lines before finding closing token: "{block_identifier} - end".')
        else:
            remaining_lines.append(line)

    if current_block is not None:
        raise ValueError(f'Formatting issue with {current_block}: reached the end of the input lines before finding closing token: "{block_identifier} - end".')

    return remaining_lines, dictionary



def add_full_reference(df, valid_index_checker):
    """
    This function adds a 'full_reference' column to a DataFrame. The 'full_reference' is calculated based on
    the values in the 'Indent' and 'Reference' columns using the following logic:

    1) If 'Indent' is 0:
       - and 'Reference' is not blank, 'full_reference' is 'Reference';
       - and 'Reference' is blank, 'full_reference' is the 'Reference' from the last row with 'Indent' 0
         that has a non-blank 'Reference'.

    2) If 'Indent' is > 0:
       - Calculate a stub reference which is equal to the value in 'Reference' enclosed in round brackets if there
         is a value in 'Reference'; otherwise, it is the 'Reference' from the last row with the same 'Indent'
         value that has a non-blank 'Reference', enclosed in round brackets.
       - Calculate the root reference which is the 'full_reference' from the last row with 'Indent' equal to
         this row's 'Indent' - 1.
       - 'full_reference' is a concatenation of the root reference and the stub reference.
    """
    df['full_reference'] = ''

    for i in range(df.shape[0]):
        # if df.loc[i, 'Text'].startswith("give effect to the constitutional right"):
        #     print('here')
        if df.loc[i, 'Indent'] == 0:
            if pd.notna(df.loc[i, 'Reference']) and df.loc[i, 'Reference'].strip() != '':
                #df.loc[i, 'full_reference'] = '(' + df.loc[i, 'Reference'].strip() + ')'
                df.loc[i, 'full_reference'] = df.loc[i, 'Reference'].strip()
            else:
                ref = df.loc[:i, :].loc[(df['Indent'] == 0) & (df['Reference'].str.strip() != ''), 'Reference'].values
                df.loc[i, 'full_reference'] = ref[-1] if ref.size > 0 else ''
        else:
            if pd.notna(df.loc[i, 'Reference']) and df.loc[i, 'Reference'].strip() != '':
                stub = df.loc[i, 'Reference']
            else:
                ref = df.loc[:i, :].loc[(df['Indent'] == df.loc[i, 'Indent']) & (df['Reference'].str.strip() != ''), 'Reference'].values
                stub = ref[-1] if ref.size > 0 else ''

            previous_full_reference = df.loc[i-1, 'full_reference']
            previous_full_reference_array = valid_index_checker.split_reference(previous_full_reference)

            if len(previous_full_reference_array) > 0:
                current_index = df.loc[i, 'Indent']

                filtered_references = []
                filtered_references.append(previous_full_reference_array[0]) # must have first index
                counter = 1
                for j in range(1, current_index):
                    if counter < len(previous_full_reference_array) and re.match(valid_index_checker.index_patterns[j], previous_full_reference_array[counter]):
                        filtered_references.append(previous_full_reference_array[counter])
                        counter += 1

                full_reference = ''.join(filtered_references) + stub
            else:
                full_reference = stub


            if not valid_index_checker.is_valid_reference(full_reference):
                raise ValueError(f'Unable to construct a valid full reference for {full_reference} which is line: {df.loc[i, "Text"]}')

            df.loc[i, 'full_reference'] = full_reference


# Note: This method will not work correctly if empty values in the dataframe are NaN as is the case when loading
#       a dataframe form a file without the 'na_filter=False' option. You should ensure that the dataframe does
#       not have any NaN value for the text fields. Try running df.isna().any().any() as a test before you get here
def get_regulation_detail(node_str, df, valid_index_tracker):
    text = ''
    terminal_text_df = df[df['full_reference'].str.startswith(node_str)]
    if len(terminal_text_df) == 0:
        return f"No section could be found with the reference {node_str}"
    terminal_text_index = terminal_text_df.index[0]
    terminal_text_indent = 0 # terminal_text_df.iloc[0]['Indent']
    for index, row in terminal_text_df.iterrows():
        number_of_spaces = (row['Indent'] - terminal_text_indent) * 4
        #set the string "line" to start with the number of spaces
        line = " " * number_of_spaces
        if pd.isna(row['Reference']) or row['Reference'] == '':
            line = line + row['Text']
        else:
            if pd.isna(row['Text']):
                line = line + row['Reference']
            else:
                line = line + row['Reference'] + " " + row['Text']
        if text != "":
            text = text + "\n"
        text = text + line

    if node_str != '': #i.e. there is a parent
        parent_reference = valid_index_tracker.get_parent_reference(node_str)
        all_conditions = ""
        all_qualifiers = ""
        while parent_reference != "":
            parent_text_df = df[df['full_reference'] == parent_reference]
            conditions = ""
            qualifiers = ""
            for index, row in parent_text_df.iterrows():
                if index < terminal_text_index:
                    number_of_spaces = (row['Indent'] - terminal_text_indent) * 4
                    if conditions != "":
                        conditions = conditions + "\n"
                    conditions = conditions + " " * number_of_spaces
                    if (row['Reference'] == ''):
                        conditions = conditions + row['Text']
                    else:
                        conditions = conditions + row['Reference'] + " " +  row['Text']
                else:
                    number_of_spaces = (row['Indent'] - terminal_text_indent) * 4
                    if (qualifiers != ""):
                        qualifiers = qualifiers + "\n"
                    qualifiers = qualifiers + " " * number_of_spaces
                    if (row['Reference'] == ''):
                        qualifiers = qualifiers + row['Text']
                    else:
                        qualifiers = qualifiers + row['Reference'] + " " + row['Text']

            if conditions != "":
                all_conditions = conditions + "\n" + all_conditions
            if qualifiers != "":
                all_qualifiers = all_qualifiers + "\n" + qualifiers
            parent_reference = valid_index_tracker.get_parent_reference(parent_reference)

        if all_conditions != "":
            text = all_conditions +  text
        if all_qualifiers != "":
            text = text + all_qualifiers

    return text

# Used to extract information for the text that contain "Chapter", "Part" or "Condition" headings
def strip_number_and_name(special_heading, string):
    special_heading_match = re.match(special_heading, string)
    remaining_string = string[special_heading_match.end():]
    number_match = re.search(r"\.", remaining_string)
    heading = remaining_string[number_match.end()+1:]
    number = remaining_string[:number_match.start()]
    return number, heading


def read_processed_regs_into_dataframe(file_list, valid_index_checker, non_text_labels, print_summary = False):
    df, non_text = process_regulations(file_list, valid_index_checker, non_text_labels)
    #TODO: Remove the page numbers from the non-text keys


    # Remove any value in the "reference" column in rows that contain "Chapter", "Part" or "Condition" section
    for i in range(df.shape[0]):
        if df.loc[i, 'Indent'] == 0:
            special_headings =  ["CHAPTER ", "Part ", "Condition "]
            if any(df.loc[i, 'Text'].startswith(special_heading) for special_heading in special_headings):
                df.at[i, 'full_reference'] = ""
            elif df.loc[i, 'Reference'] != '':
                df.loc[i, 'Heading'] = True

    # Remove any value in the "full_reference" column in rows that contain "Chapter", "Part" or "Condition" section so
    #   it can be deleted later
    # Move the Chapter, Part and Condition lines to columns
    chapter_number = ""
    chapter_name = ""
    part_number = ""
    part_name = ""
    condition_number = ""
    condition_name = ""

    for i in range(df.shape[0]):
        if df.loc[i, 'Indent'] == 0:
            if df.loc[i, 'Text'].startswith("CHAPTER "):
                df.at[i, 'full_reference'] = ""
                chapter_number, chapter_name = strip_number_and_name("CHAPTER ", df.loc[i, 'Text'])
                part_number = ""
                part_name = ""
                condition_number = ""
                condition_name = ""

            elif df.loc[i, 'Text'].startswith("Part "):
                df.at[i, 'full_reference'] = ""
                part_number, part_name = strip_number_and_name("Part ", df.loc[i, 'Text'])
                condition_number = ""
                condition_name = ""

            elif df.loc[i, 'Text'].startswith("Condition "):
                df.at[i, 'full_reference'] = ""
                condition_number, condition_name = strip_number_and_name("Condition ", df.loc[i, 'Text'])

            if df.loc[i, 'Reference'] != '':
                df.loc[i, 'Heading'] = True

        df.loc[i, 'chapter_no'] = chapter_number
        df.loc[i, 'chapter_heading'] = chapter_name
        df.loc[i, 'part_no'] = part_number
        df.loc[i, 'part_heading'] = part_name
        df.loc[i, 'condition_no'] = condition_number
        df.loc[i, 'condition_heading'] = condition_name

    # now drop the rows that have no full_reference
    df = df[df["full_reference"] != ""]

    if print_summary:
        print("total lines in dataframe: ", len(df))
        for key in non_text.keys():
            print("total ", key, ": ", len(non_text[key]))

    return df, non_text


