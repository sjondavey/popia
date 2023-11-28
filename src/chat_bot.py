import logging
import pandas as pd
from openai import OpenAI
from collections import Counter
import os
import fnmatch
import regex # fuzzy lookup of references in a section of text

import importlib
import src.valid_index
importlib.reload(src.valid_index)
from src.valid_index import ValidIndex, get_popia_act_index

import src.file_tools
importlib.reload(src.file_tools)
from src.file_tools import get_regulation_detail
                           

import src.embeddings
importlib.reload(src.embeddings)
from src.embeddings import get_ada_embedding, \
                           get_closest_nodes, \
                           num_tokens_from_string,  \
                           num_tokens_from_messages

client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
)

class PopiaAct():
    def __init__(self, 
                 path_to_manual_as_csv_file,
                 path_to_definitions_as_parquet_file,
                 path_to_index_as_parquet_file,
                 log_file = '', 
                 logging_level = 20): # 20 = logging.info This will exclude my DEV_LEVEL labeled logs

        # Create a custom log level for the really detailed logs
        self.DEV_LEVEL = 15
        logging.addLevelName(self.DEV_LEVEL, 'DEV')        

        # Set up basic configuration first
        if log_file == '':
            logging.basicConfig(level=logging_level)
        else: 
            logging.basicConfig(filename=log_file, filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging_level)
        
        # Then get the logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging_level)

        self.user_type = "Responsible Party" 
        self.manual_name = "\'Protection of Personal Information Act\' (POPIA)"

        self.index_checker = get_popia_act_index()
        if os.path.exists(path_to_manual_as_csv_file):
            self.df_popia = pd.read_csv(path_to_manual_as_csv_file, sep="|", encoding="utf-8", na_filter=False)  
            if self.df_popia.isna().any().any():
                msg = f'Encountered NaN values while loading {path_to_manual_as_csv_file}. This will cause ugly issues with the get_regulation_detail method'
                logging.error(msg)
                raise ValueError(msg)
        else:
            msg = f"Could not find the file {path_to_manual_as_csv_file}"
            logging.error(msg)
            raise FileNotFoundError(msg)
        
        # Load the definitions. 
        if os.path.exists(path_to_definitions_as_parquet_file):
            self.df_definitions_all = pd.read_parquet(path_to_definitions_as_parquet_file, engine='pyarrow')
        else:
            msg = f"Could not find the file {path_to_definitions_as_parquet_file}"
            logging.error(msg)
            raise FileNotFoundError(msg)
        
        # Load the section headings. 
        if os.path.exists(path_to_index_as_parquet_file):
            self.df_text_all = pd.read_parquet(path_to_index_as_parquet_file, engine='pyarrow')
        else:
            msg = f"Could not find the file {path_to_index_as_parquet_file}"
            logging.error(msg)
            raise FileNotFoundError(msg)

        self.system_states = ["rag",                          # System is going to try RAG         
                              "no_relevant_embeddings",       # found embeddings but LLM doesn't think they help answer the question
                              "requires_additional_sections", # System asked for something but we could not identify it as a valid section OR a query to the text with this index returns null
                              "stuck"]                        # Terminal state. Once in this state no further progress can be made and user must restart
        self.system_state = self.system_states[0]

        
        self.messages = []

        self.rag_prefixes = ["ANSWER:",  # LLM was able to answer the question with the input data
                             "SECTION:", # LLM Requested additional information to answer the question
                             "NONE:",    # The input data was not helpful
                             "FAIL:"]    # The LLM did not follow instructions

        self.assistant_msg_no_data = "I was unable to find any relevant documentation to assist in answering the question. Can you try re-phrasing the question?"
        self.assistant_msg_no_relevant_data = "The documentation I have been provided does not help me answer the question. Please re-phrase it and lets try again?"
        self.assistant_msg_stuck = "Unfortunately the system is in an unrecoverable state. Please restart the chat"
        self.assistant_msg_unknown_state = "The system is in an unknown state and cannot proceed. Please restart the chat"
        self.assistant_msg_llm_not_following_instructions = "The call to the LLM resulted in a response that did not fit parameters, even after retrying it. Please restart the chat and try phrasing the question differently"



    def reset_conversation_history(self):
        # opening_message = f"I am a bot designed to answer questions based on the {self.manual_name}. How can I assist today?"
        # self.messages = [{"role": "assistant", "content": opening_message}]
        self.messages = []
        self.system_state = self.system_states[0]
        
    # Note: To test the workflow I need some way to control the openai API responses. I have chosen to do this with the two parameters
    #       testing: a flag. If false the function will run calling the openai api for responses. If false the function will 
    #                        select the response from the list of responses manual_responses_for_testing
    #       manual_responses_for_testing: A list of text. If testing == True, these values will be used as if they were the 
    #                                     the response from the API. This function can make multiple calls to the API so the i-th
    #                                     row in the list corresponds to the i-th call of the API
    #  
    def user_provides_input(self, user_context, threshold, model_to_use, temperature, max_tokens,
                            testing = False, manual_responses_for_testing = []):
        
        if user_context is None:
            self.logger.error("The user passed an empty string. This should not have happened and is an indication there is a bug in the frontend. The system will be placed into a 'stuck' status")
            self.messages.append({"role": "assistant", "content": self.assistant_msg_unknown_state})
            self.system_state == self.system_states[3] #stuck
            return

        # I use this variable if I ever need to remove the RAG decoration from the user question
        #self.user_question = user_context

        if len(self.messages) == 0:
            self.messages.append({"role": "user", "content": user_context})
        elif not (self.messages[-1]["role"] == "user" and self.messages[-1]["content"] == user_context): 
            self.messages.append({"role": "user", "content": user_context})
        # else the message is already in the queue

        if self.system_state == self.system_states[3]: #stuck
            self.messages.append({"role": "assistant", "content": self.assistant_msg_stuck})
            return 
        elif self.system_state == self.system_states[0]: #"rag":
            self.logger.info(f"User Question:\n{user_context}")
            df_definitions, df_search_sections = self.similarity_search(user_context, threshold)
            if len(self.messages) < 2 and (len(df_definitions) + len(df_search_sections) == 0):
                self.logger.log(self.DEV_LEVEL, "Unable to find any definitions or text related to this query")
                self.system_state = self.system_states[0] # "rag"
                self.messages.append({"role": "assistant", "content": self.assistant_msg_no_data})
                self.logger.info(f"Assistant:\n{self.assistant_msg_no_data}")
                return
            else:
                flag, response = self.resource_augmented_query(model_to_use, temperature, max_tokens, df_definitions, df_search_sections,
                                                               testing, manual_responses_for_testing)
                if flag == self.rag_prefixes[0]: # "ANSWER:"
                    self.messages.append({"role": "assistant", "content": response.strip()})
                    self.logger.info(f"assistant:\n {response.strip()}")
                    self.system_state = self.system_states[0] # RAG
                    return 
                elif flag == self.rag_prefixes[1]: # "SECTION:"
                    self.logger.log(self.DEV_LEVEL, f"Question asked. Request for more info:\n{response}")
                    modified_section_to_add = self.index_checker.extract_valid_reference(response)
                    if modified_section_to_add is None:
                        # TODO: Do you want to ask the user for help?
                        self.logger.info("Request to add resources failed because the reference was not valid")
                        self.system_state = self.system_states[3] # Stuck
                        self.messages.append({"role": "assistant", "content": self.assistant_msg_stuck}) 
                        return                    
                    df_search_sections = self.add_section_to_resource(modified_section_to_add, df_search_sections)
                    if self.system_state == self.system_states[2]: # "requires_additional_sections"
                        # TODO: Do you want to ask the user for help?
                        self.logger.info("Request to add resources failed")
                        self.system_state = self.system_states[3] # Stuck
                        self.messages.append({"role": "assistant", "content": self.assistant_msg_stuck}) 
                        return

                    # Remove the RAG decoration from the user question
                    if self.messages[-1]["role"] != "user":
                        raise AttributeError("There is a problem with the logic. The last message on the queue must be from the user")
                    self.messages[-1]["content"] = user_context
                    # try again with new resources
                    flag, response = self.resource_augmented_query(model_to_use, temperature, max_tokens, df_definitions, df_search_sections,
                                                                   testing, manual_responses_for_testing[1:])
                    if flag == self.rag_prefixes[0]: # "ANSWER:"
                        self.logger.log(self.DEV_LEVEL, "Question answered with the additional information")
                        self.messages.append({"role": "assistant", "content": response.strip()})
                        self.system_state = self.system_states[0] # RAG
                        self.logger.info(f"assistant:\n {response.strip()}")
                        return
                    else: 
                        self.logger.log(self.DEV_LEVEL, "Even with the additional information, they system was unable to answer the question. Placing the system in 'stuck' mode")
                        msg = "A call for additional information to answer the question failed. The system is now stuck. Please restart it"
                        self.messages.append({"role": "assistant", "content": msg})                        
                        self.system_state = self.system_states[3] # Stuck
                        self.logger.info(f"assistant:\n {msg}")                        
                        return

                elif flag == self.rag_prefixes[2]: # "NONE:"
                    self.logger.log(self.DEV_LEVEL, "The LLM was not able to find anything relevant in the supplied sections")
                    self.messages.append({"role": "assistant", "content": self.assistant_msg_no_relevant_data})
                    self.system_state = self.system_states[0] # RAG
                    self.logger.info(f"assistant:\n {self.assistant_msg_no_relevant_data}")                        
                    return

                else:
                    self.logger.error("RAG returned an unexpected response")
                    self.messages.append({"role": "assistant", "content": self.assistant_msg_llm_not_following_instructions})
                    self.system_state = self.system_states[3] #stuck # We are at a dead end.
                    self.logger.info(f"assistant:\n {self.assistant_msg_llm_not_following_instructions}")                        
                    return
        else:
            self.logger.error("The system is in an unknown state")
            self.messages.append({"role": "assistant", "content": self.assistant_msg_unknown_state})
            return

    def add_section_to_resource(self, section_to_add, df_search_sections):
        # Step 1) confirm it is requesting something that passes validation
        modified_section_to_add = self.index_checker.extract_valid_reference(section_to_add)
        if modified_section_to_add is None:
            self.system_state = self.system_states[2] # "requires_additional_sections"
            return df_search_sections
        try: # passes index verification but there is an error retrieving the section
            self.get_regulation_detail(modified_section_to_add)
        except Exception as e:
            self.system_state = self.system_states[2] # "requires_additional_sections"
            return df_search_sections
          
        referring_sections = self._find_reference_that_calls_for(modified_section_to_add, df_search_sections)
        
        if len(referring_sections) > 0: # Delete the other sections, keep the referring section and the new data
            referring_sections.append(modified_section_to_add)
            # now create the new RAG df_search_sections
            manual_data = []
            for i in range(len(referring_sections)):
                section = referring_sections[i]
                count = 1
                raw_text = self.get_regulation_detail(section)
                token_count = num_tokens_from_string(raw_text)
                manual_data.append([section, 1.0, count, raw_text, token_count])            
            df_manual_data = pd.DataFrame(manual_data, columns = ["reference", "cosine_distance", "count", "raw_text", "token_count"])
            return df_manual_data
        else: # Just add the new data and hope the total context is not too long
            section = modified_section_to_add
            count = 1
            raw_text = self.get_regulation_detail(section)
            token_count = num_tokens_from_string(raw_text)
            df_to_add = pd.DataFrame([[section, 1.0, count, raw_text, token_count]], columns = ["reference", "cosine_distance", "count", "raw_text", "token_count"])
            df_search_sections = pd.concat([df_search_sections, df_to_add]).reset_index(drop=True)
            #df_search_sections.loc[len(df_search_sections.index)] = [section, 1.0, count, raw_text, token_count]
            return df_search_sections


    # Note that 'section' is assumed to be valid at this stage
    def _find_reference_that_calls_for(self, valid_section_index, df_search_sections):
        referring_section = []
        if len(df_search_sections) == 0:
            return referring_section
        if len(df_search_sections) == 1:
            referring_section.append(df_search_sections.iloc[0]["reference"])
        else:
            for index, row in df_search_sections.iterrows():
                match = self._find_fuzzy_reference(row["raw_text"], valid_section_index)
                if match:
                    referring_section.append(row["reference"])
        if len(referring_section) == 0:
            self.logger.error(f"The LLM asked for an additional valid reference {valid_section_index} but we could not determine which section referred to it")
        return referring_section
       

    # TODO: Think about replacing this with just the function ValidIndex.extract_valid_reference
    # I think that if we delete the first line of raw_text then we should be able to run ValidIndex.extract_valid_reference
    # on the remaining text to see if the valid_section_index was in the raw_section. It will get rid of some code and cater
    # also for cases with the reference in the raw_section is not correctly formatted rather than allowing for some random 
    # number of mismatches as I do here
    def _find_fuzzy_reference(self, raw_section, valid_section_index):
        # Enabling fuzzy matching with 2 insertions/deletions/substitutions to cater for stray spaces that may creep into the text
        pattern = r'(%s){e<=2}' % regex.escape(valid_section_index)
        match = regex.search(pattern, raw_section)
        if match:
            return match.group()
        else:
            return None

    def _create_system_message(self):
        return f"You are answering questions for an {self.user_type} based only on the relevant sections from the {self.manual_name} that are provided. You have three options:\n\
1) Answer the question. Preface an answer with the tag '{self.rag_prefixes[0]}'. If possible, end the answer with the reference to the section or sections you used to answer the question.\n\
2) Request additional documentation. If, in the body of the sections provided, there is a reference to another section of the Manual that is directly relevant and not already provided, respond with the word '{self.rag_prefixes[1]}' followed by the full section reference.\n\
3) State '{self.rag_prefixes[2]}' and nothing else in all other cases\n\n\
Note: In the manual sections are numbered like 100.(a) or 47.(1)(a). The first index uses the regex pattern r'(?:[1-9]|[1-9][0-9]|[1-9][0-9]{2})\.'. The second is a number in round brackets which may or may not be there. Thereafter, each sub-index is surrounded by round brackets"

    def _add_rag_data_to_question(self, question, df_definitions, df_search_sections):
        user_context = f'Question: {question}\n\n'
        if len(df_definitions) > 0:
            user_context = user_context + "Definitions from the Manual\n"
            for index, row in df_definitions.iterrows():
                user_context = user_context + f"{row['Definition']}\n"
        if len(df_search_sections) > 0:
            user_context = user_context + "Sections from the Manual\n"
            for index, row in df_search_sections.iterrows():
                user_context = user_context + f"{row['raw_text']}\n"
        return user_context

    def _extract_question_from_rag_data(self, decorated_question):
        return decorated_question.split("\n")[0][len("Question: "):]

    # Make sure the last entry in self.messages is the user context
    # Note: To test the workflow I need some way to control the responses. I have chosen to do this with the two parameters
    #       testing: a flag. If false the function will run calling the openai api for responses. If false the function will 
    #                        select the response from the list of responses manual_responses_for_testing
    #       manual_responses_for_testing: A list of text. If testing == True, these values will be used as if they were the 
    #                                     the response from the API. This function can make multiple calls to the API so the i-th
    #                                     row in the list corresponds to the i-th call of the API
    #  
    def resource_augmented_query(self, model_to_use, temperature, max_tokens, df_definitions, df_search_sections,
                                 testing = False, manual_responses_for_testing = []):
        if len(self.messages) == 0 or self.messages[-1]["role"] != "user": 
            self.logger.error("resource_augmented_query method called but the last message on the stack was not from the user")
            self.system_state == self.system_states[3] # stuck
            return self.system_states[3], self.system_states[3]
        if self.system_state != "rag":
            self.logger.error("resource_augmented_query method called but the the system is not in rag state")
            self.system_state == self.system_states[3] # stuck
            return self.system_states[3], self.system_states[3]
        

        if len(self.messages) > 1 or len(df_definitions) + len(df_search_sections) > 0: # should always be the case as we check this in the control loop
            self.logger.log(self.DEV_LEVEL, "#################   RAG Prompts   #################")

            system_content = self._create_system_message()
            self.logger.info("System Prompt:\n" + system_content)

            # Replace the user question with the RAG version of it
            user_question = self.messages[-1]["content"]
            if user_question.startswith("Question:"):
                user_question = self._extract_question_from_rag_data(user_question)
            self.messages[-1]["content"] = self._add_rag_data_to_question(user_question, df_definitions, df_search_sections)
            self.logger.info("User Prompt with RAG:\n" + self.messages[-1]["content"])


            # Create a temporary message list. We will only add the messages to the chat history if we get well formatted answers
            system_message = [{"role": "system", "content": system_content}]
            truncated_chat = self._truncate_message_list(system_message, self.messages, 2000)
            # if len(self.messages) != len(truncated_chat):
            #     self.logger.info(f"Total message queue contains {len(self.messages)} messages but we have truncated these to just the last {len(truncated_chat)}")


            if testing == True:
                self.logger.log(self.DEV_LEVEL, "Using canned answers rather than making calls to the openai API")
                initial_response = manual_responses_for_testing[0]
            else:
                total_tokens = num_tokens_from_messages(truncated_chat, model_to_use)
                if (model_to_use == "gpt-3.5-turbo" or model_to_use == "gpt-4") and total_tokens > 3500 and model_to_use!="gpt-3.5-turbo-16k":
                    self.logger.warning("!!! NOTE !!! You have a very long prompt. Switching to the gpt-3.5-turbo-16k model")
                    model_to_use = "gpt-3.5-turbo-16k"

                response = client.chat.completions.create(
                                    model=model_to_use,
                                    temperature = temperature,
                                    max_tokens = max_tokens,
                                    messages = truncated_chat
                                )
                #print(response)
                initial_response = response.choices[0].message.content

            # Check the model performed as instructed prefacing its response with one of three words 
            for prefix in self.rag_prefixes:
                if initial_response.startswith(prefix):
                    # self.messages[-1]["content"] = self.user_question
                    return prefix, initial_response[len(prefix):]

            # The model did not perform as instructed so we not ask it to check its work
            self.logger.warning("Initial chat API response did not result in a response with the correct format. Retrying")
            self.logger.warning(f"The response was:\n{initial_response}")

            despondent_user_context = f"Please check your answer and make sure you preface your response using only one of the three permissible words, {self.rag_prefixes[0]}, {self.rag_prefixes[1]} or {self.rag_prefixes[2]}"
            despondent_user_messages = truncated_chat + [
                                        {"role": "assistant", "content": initial_response},
                                        {"role": "user", "content": despondent_user_context}]
                                        
            if testing == True and len(manual_responses_for_testing) > 1:
                self.logger.log(self.DEV_LEVEL, "Using canned answers rather than making calls to the openai API")
                followup_response_text = manual_responses_for_testing[1]
            else:
                total_tokens = num_tokens_from_messages(despondent_user_messages, model_to_use)
                if (model_to_use == "gpt-3.5-turbo" or model_to_use == "gpt-4") and total_tokens > 3500 and model_to_use!="gpt-3.5-turbo-16k":
                    self.logger.warning("!!! NOTE !!! You have a very long prompt. Switching to the gpt-3.5-turbo-16k model")
                    model_to_use = "gpt-3.5-turbo-16k"

                followup_response = client.chat.completions.create(
                                    model=model_to_use,
                                    temperature = temperature,
                                    max_tokens = max_tokens,
                                    messages = despondent_user_messages
                                )
                followup_response_text = followup_response.choices[0].message.content
            for prefix in self.rag_prefixes:
                if followup_response_text.startswith(prefix):
                    # self.messages[-1]["content"] = self.user_question
                    return prefix, followup_response_text[len(prefix):]

        return self.rag_prefixes[3], "The LLM was not able to return an acceptable answer. "
        

    def similarity_search(self, user_context, threshold = 0.15):
        question_embedding = get_ada_embedding(user_context)        
        self.logger.log(self.DEV_LEVEL, "#################   Similarity Search       #################")
        relevant_definitions = get_closest_nodes(self.df_definitions_all, embedding_column_name = "Embedding", question_embedding = question_embedding, threshold = threshold)
        if len(relevant_definitions) > 0:
            self.logger.log(self.DEV_LEVEL, "--   Relevant Definitions")
            for index, row in relevant_definitions.iterrows():
                self.logger.log(self.DEV_LEVEL, f'{row["cosine_distance"]:.4f}: ({row["source"]:>10}): {row["Definition"]}')
        else:
            self.logger.log(self.DEV_LEVEL, "--   No relevant definitions found")

        relevant_sections = get_closest_nodes(self.df_text_all, embedding_column_name = "Embedding", question_embedding = question_embedding, threshold = threshold)
        if len(relevant_sections) > 0:
            max_search_items = 15
            if len(relevant_sections) > max_search_items: # sometime the search would return a lot of values and this leads to some issues when looking at sections that get referenced often
                self.logger.log(self.DEV_LEVEL, f"Found more than {max_search_items} references that are closer than the input threshold of {threshold}. Capping them so there are only {max_search_items}")
                relevant_sections = relevant_sections.nsmallest(max_search_items, 'cosine_distance')            # 1) The top result

            self.logger.log(self.DEV_LEVEL, "--   Relevant Sections")
            for index, row in relevant_sections.iterrows():
                self.logger.log(self.DEV_LEVEL, f'{row["cosine_distance"]:.4f}: {row["section"]:>20}: {row["source"]:>15}: {row["text"]}')

            filtered_relevant_sections = self._filter_relevant_sections(relevant_sections)
            self.logger.log(self.DEV_LEVEL, "--   Filtered Sections")
            for index, row in filtered_relevant_sections.iterrows():
                if row["count"] == 1:
                    self.logger.log(self.DEV_LEVEL, f'{row["cosine_distance"]:.4f}            : {row["reference"]:>20}: {row["count"]:>2}')                
                else:
                    self.logger.log(self.DEV_LEVEL, f'{row["cosine_distance"]:.4f} (*min dist): {row["reference"]:>20}: {row["count"]:>2}')                
            filtered_relevant_sections["raw_text"] = filtered_relevant_sections["reference"].apply(self.get_regulation_detail)
            filtered_relevant_sections["token_count"] = filtered_relevant_sections["raw_text"].apply(num_tokens_from_string)

            # max out at 5 pieces of data
            if len(filtered_relevant_sections) > 5:
                self.logger.log(self.DEV_LEVEL, "Truncating the number of documents to 5")

            sorted_df = filtered_relevant_sections.sort_values(by="cosine_distance", ascending=True).head(5)

            return relevant_definitions, sorted_df 
        else:
            self.logger.log(self.DEV_LEVEL, "--   No relevant sections found")
            return relevant_definitions, pd.DataFrame([], columns = ["reference", "cosine_distance", "count", "raw_text", "token_count"]) 
        

    # Logic to refine the choice of data that will be sent to the LLM
    def _filter_relevant_sections(self, relevant_sections):
        # Get the top result
        search_sections = []
        if len(relevant_sections) > 0:     
            # 1) The top result
            top_result = relevant_sections.iloc[0]["section"]
            cosine_distance = relevant_sections.iloc[0]["cosine_distance"]
            count = len(relevant_sections[relevant_sections['section'] == top_result])
            search_sections.append([top_result, cosine_distance, count])
            self.logger.log(self.DEV_LEVEL, f'Top result: {top_result} with a cosine distance of {cosine_distance:.4f}')

            # 2) The mode
            mode_value_list = relevant_sections['section'].mode()
            mode_value = ""
            if len(mode_value_list) == 1:
                mode_value = mode_value_list[0]
                mode_already_in_search_sections = any(mode_value == sublist[0] for sublist in search_sections) # check if the mode_value appears in the first column of search_sections
                if not mode_already_in_search_sections: # i.e. it is not also the top result
                    sub_frame = relevant_sections[relevant_sections['section'] == mode_value]
                    count = len(sub_frame)
                    minimum_cosine_distance = sub_frame['cosine_distance'].min()
                    search_sections.append([mode_value, minimum_cosine_distance, count])
                    self.logger.log(self.DEV_LEVEL, f"Most common section: {mode_value} with a minimum cosine distance of {minimum_cosine_distance}")
            else:
                self.logger.log(self.DEV_LEVEL, "No unique mode")

            # 3) References that are found frequently that are not the mode 
            count_dict = Counter(relevant_sections['section'])
            repeated_items = {k: v for k, v in count_dict.items() if v > 1}
            # Remove the top value and mode from repeated_items if it exists
            if top_result in repeated_items:
                del repeated_items[top_result]
            if mode_value in repeated_items:
                del repeated_items[mode_value]        
            if (len(repeated_items) > 0):
                self.logger.log(self.DEV_LEVEL, "References found that occur multiple times")
                for reference, count in repeated_items.items():
                    sub_frame = relevant_sections[relevant_sections['section'] == reference]
                    count = len(sub_frame)
                    minimum_cosine_distance =  sub_frame['cosine_distance'].min()
                    search_sections.append([reference, minimum_cosine_distance, count])
                    self.logger.log(self.DEV_LEVEL, f"Reference: {reference}, Count: {count}, Min Cosine-Distance: {minimum_cosine_distance}")

        if len(search_sections) == 1 and len(relevant_sections) > 1: # the case if each section only appears once in the search
            # remove the top search result from the list
            remaining_relevant_sections = relevant_sections[relevant_sections["section"] != search_sections[0][0]]
            self.logger.log(self.DEV_LEVEL, f'Only the top result added but more were found. Adding the next most likely answer')
            if len(remaining_relevant_sections) >= 1:
                second_result = remaining_relevant_sections.iloc[0]
                search_sections.append([second_result['section'], second_result['cosine_distance'], 1])
            if len(remaining_relevant_sections) >= 2:
                third_result = remaining_relevant_sections.iloc[1]
                search_sections.append([third_result['section'], third_result['cosine_distance'], 1])

        # Note the order of the search_section is preserved        
        return pd.DataFrame(search_sections, columns=["reference", "cosine_distance", "count"])


    def get_regulation_detail(self, node_str):
        valid_reference = self.index_checker.extract_valid_reference(node_str)
        if not valid_reference:
            return "The reference did not conform to this documents standard"
        else:
            return get_regulation_detail(valid_reference, self.df_popia, self.index_checker)

    # will return a minimum message that contains the system message and the last message in the message list even if this is longer than the token_limit
    def _truncate_message_list(self, system_message, message_list, token_limit = 2000):
        if len(message_list) == 0:
            return system_message
        token_count = num_tokens_from_string(system_message[0]["content"]) + num_tokens_from_string(message_list[-1]["content"])
        number_of_messages = 1
        while number_of_messages < len(message_list) and token_count < token_limit:
            number_of_messages += 1
            token_count += num_tokens_from_string(message_list[-number_of_messages]["content"])
        number_of_messages = max(1, number_of_messages - 1)
        truncated_messages = []
        truncated_messages.append(system_message[0])
        for msg in message_list[-number_of_messages:]:
            truncated_messages.append(msg)
        return truncated_messages


        # if len(message_list) < 4: # user message + previous response + previous user message
        #     return message_list
        # else:
        #     n = 3
        #     token_count = num_tokens_from_string(message_list[-1]["content"]) + \
        #                   num_tokens_from_string(message_list[-2]["content"]) + \
        #                   num_tokens_from_string(message_list[-3]["content"])
        #     while token_count < token_limit and len(message_list) > n:
        #         token_count += num_tokens_from_string(message_list[-n]["content"])
        #         n += 1
        # if n > 3:
        #     return message_list[-(n-1):]
        # else:
        #     return message_list[-3:]


