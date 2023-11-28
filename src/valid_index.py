import re

# Legal documents are often written where every line is given a reference in a tree structure. The numbering follows a format
# where for example you start with a Capital Letter, then an indented Roman Numeral, then a double indented lowercase letter
# in brackets etc etc.
# This is a helper class for indexes like this. It also has a list of exclusions because these texts often start with a section
# or two that do not follow the default numbering
class ValidIndex():
    def __init__(self, regex_list_of_indices, exclusion_list = []):
        self.index_patterns = regex_list_of_indices
        self.exclusion_list = exclusion_list

    # Note: This does check the order
    def is_valid_reference(self, reference):
        if reference in self.exclusion_list:
            return True

        reference_copy = reference
        pattern_matched = False

        # first pattern must be there - first
        match = re.match(self.index_patterns[0], reference_copy)
        if match:
            reference_copy = reference_copy[match.end():]
            pattern_matched = True
        else:
            return False

        if len(reference_copy) > 0:
            for pattern in self.index_patterns[1:]:
                if (len(reference_copy) > 0):
                    match = re.match(pattern, reference_copy)
                    if match:
                        reference_copy = reference_copy[match.end():]
                        pattern_matched = True
                        #reference = reference[match.end():]
                    else:
                        pattern_matched = False
        return pattern_matched

    # When dealing with text (and the output from an LLM) we often get something that is supposed to be a 
    # reference but has other text or characters interspersed. This is an attempt to extract a valid reference
    # from text that contains a valid reference plus potentially some 'other stuff'. It works by
    # - Extracting the i-ith index_patterns. 
    # - appending this to a string called partial_ref 
    # - removing the matched string plus any other text to the left of it 
    # - next i
    # The method returns None if it does not find the i-th pattern and there is still text remaining to search 

    # For example, give the excon reference pattern, this method will provide the following output
    # print(extract_valid_reference('B.18 Gold (B)(i)(b)'))  # Output: 'B.18(B)(i)(b)'
    # print(extract_valid_reference('B.18 Gold (B)(a)(b)'))  # Output: None because after (B) we need a roman numeral
    # print(extract_valid_reference('A.1'))  # Output: 'A.1'
    def extract_valid_reference(self, input_string):
        if input_string.strip() in self.exclusion_list:
            return input_string.strip()

        partial_ref = ""
        remaining_str = input_string
    
        for pattern in self.index_patterns:
            if pattern[0] == "^": # the caret "^" is used in the index pattern because we only want the index at the start of the section but this causes potential issues here so it is removed 
                pattern = pattern[1:]
            match = re.search(pattern, remaining_str)
            if match:
                partial_ref += match.group()
                remaining_str = remaining_str[match.end():]
            else:
                if remaining_str and "(" in remaining_str:                     
                    #return None # there is still some text left and because it contains an "(" is "should" be part of the reference
                    return partial_ref # this will deal with some cases but may result in undesired behaviour for invalid strings of the form ('B.18 Gold (B)(a)(b)'))
        
        return partial_ref if partial_ref else None


    def split_reference(self, reference):
        components = []
        if reference == "": # i.e. the root node which has no name
            return components
        if reference in self.exclusion_list:
            components.append(reference)
            return components

        # Initialize variables
        reference_copy = reference
        pattern_matched = False

        # first pattern must be there - first
        match = re.match(self.index_patterns[0], reference_copy)
        if match:
            components.append(match.group(0))
            reference_copy = reference_copy[match.end():]
            pattern_matched = True
        else:
            raise ValueError(f'The input index {reference} does not start with the correct pattern')

        if len(reference_copy) > 0:
            for pattern in self.index_patterns:
                if (len(reference_copy) > 0):
                    match = re.match(pattern, reference_copy)
                    if match:
                        components.append(match.group(0))
                        reference_copy = reference_copy[match.end():]
                        pattern_matched = True
                    else:
                        pattern_matched = False
        # If there's anything left in the reference after all patterns have been attempted, it's invalid.
        if reference_copy or (pattern_matched == False):
            raise ValueError(f'The input index {reference} did not comply with the schema')
        return components

    def get_parent_reference(self, input_string):
        if input_string == "":
            raise ValueError(f"Unable to get parent string for empty input")
        #print(f"Calling valid_index_checker.split_reference(input_string) with intput: {input_string}")
        split_reference = self.split_reference(input_string)
        parent_reference = ''
        if len(split_reference) == 0:
            raise ValueError(f"Unable to extract valid indexes from the string {input_string}")
        else: #note this also covers the case len(split_reference) == 1
            for i in range(0, len(split_reference)-1):
                parent_reference += split_reference[i]
        return parent_reference


    # From a line of text, extract the indent (mod 4), the index and the remaining text from the line. Since the 
    # the indent (mod 4) defines the expected regex pattern for the index, we check that the index matches the appropriate
    # regex pattern
    def parse_line_of_text(self, line_of_text):
        stripped_line = line_of_text.lstrip(' ')
        indent = len(line_of_text) - len(stripped_line)
        if indent % 4 != 0:
            raise ValueError(f"This line does not have an indent which is a multiple of 4: {line_of_text}")
        indent = indent // 4
        index, remaining_text = self._extract_reference_from_string(stripped_line)
        if index != "": # check that, if there is an index, it is the correct Index type
            if index in self.exclusion_list:
                if indent == 0:
                    return indent, index, remaining_text
                else:
                    raise ValueError(f"This line has {indent} indent(s) but should have zero because the index is on the exclusion list")

            if indent >= len(self.index_patterns):
                raise ValueError(f"This line has too many indents and cannot be compared against a Valid Index: {line_of_text}")
            expected_pattern = self.index_patterns[indent]
            match = re.match(expected_pattern, index)
            if not match:
                raise ValueError(f"This line has {indent} indent(s) and its index should match a regex pattern {expected_pattern} but it does not: {line_of_text}")

        return indent, index, remaining_text

    # Note: This method only extracts things that look like a reference from the start of the string (i.e. if the string
    #       starts with a blank it will test the blank against the index). It does not perform a sanity test to see
    #       if the line with the reference is indented correctly because at this stage, with only one line of data
    #       it is impossible to know, for example if '(i)' is a roman numeral or the single lowercase letter.
    #       At this stage we just strip matching strings that look like index values from the rest of the string.
    #       We only check if (i) is a indented correctly later when we have the full reference
    #
    def _extract_reference_from_string(self, s):
        counter = 0
        for pattern in self.index_patterns:
            match = re.match(pattern, s)
            if match:
                # If a match is found, return the matched index and the remaining string
                return match.group(0), s[match.end()+1:] # there is always a space after the index
            counter += 1

        for exclusion_item in self.exclusion_list:
            if s.strip() == exclusion_item:
                return exclusion_item, ''
        # If no match is found, return an empty string for the index and the original string
        return '', s


def get_popia_act_index():
    #exclusion_list = ['Legal context', 'Introduction']
    exclusion_list = ['CHAPTER']
    index_patterns = [
        r'^(?:[1-9]|[1-9][0-9]|[1-9][0-9]{2})\.',
        r'^\([1-9]\)',
        r'^\([a-z]\)',
        r'^\((i|ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii|xiii|xiv|xv|xvi|xvii|xviii|xix|xx|xxi|xxii|xxiii|xxiv|xxv|xxvi|xxvii)\)',
        r'^\([a-z]{2}\)',
    ]
    return ValidIndex(regex_list_of_indices=index_patterns, exclusion_list=exclusion_list)

