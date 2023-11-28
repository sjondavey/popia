from anytree import Node, RenderTree, find, LevelOrderIter, AsciiStyle
import re
import pandas as pd
from src.valid_index import ValidIndex

from src.file_tools import get_regulation_detail
from src.embeddings import num_tokens_from_string
        

class TreeNode(Node):
    def __init__(self, name, full_node_name, parent=None, heading_text=''):
        super().__init__(name, parent=parent)
        self.heading_text = heading_text
        self.full_node_name = full_node_name

    # Recursive function to consolidate headings from leaves to root
    def consolidate_from_leaves(self, consolidate_headings):
        # base case: if the node is a leaf node (no children)
        if not self.children:
            return self.heading_text
        
        # Recursive case: if the node has children
        children_headings = [child.consolidate_from_leaves(consolidate_headings) for child in self.children]
        self.heading_text = consolidate_headings(children_headings)
        return self.heading_text

class Tree:
    def __init__(self, root_id, valid_index_checker):
        self.root = TreeNode(root_id, "", parent=None, heading_text='')
        self.valid_index_checker = valid_index_checker

    def add_to_tree(self, node_str, heading_text=''):
        if node_str == self.root.name:
            self.root.heading_text = heading_text
            return
        elif not self.valid_index_checker.is_valid_reference(node_str):
            raise ValueError(f'{node_str} is not a valid node reference')

        node_names = self.valid_index_checker.split_reference(node_str)
        current_parent = self.root
        full_node_name = ''
        previous_full_node_name = ''  # variable to hold previous full node name
        for i, node_name in enumerate(node_names):
            previous_full_node_name = full_node_name  # update previous node name before adding current node name
            full_node_name = full_node_name + node_name
            # Try to find the node as a child of the current parent
            found_node = None
            for child in current_parent.children:
                if child.name == node_name:
                    found_node = child
                    break
            # If the node isn't found, create it
            if found_node is None:
                if i == len(node_names) - 1:  # if this is the last node
                    current_parent = TreeNode(node_name, previous_full_node_name + node_name, parent=current_parent, heading_text=heading_text)
                else:
                    current_parent = TreeNode(node_name, previous_full_node_name + node_name, parent=current_parent, heading_text='')
            else:
                # # if found node has a heading text and it's not the last node or it's the last node and heading text is not the same, raise an exception
                # if found_node.heading_text and ((i != len(node_names) - 1) or (heading_text != '' and i == len(node_names) - 1 and found_node.heading_text != heading_text)):
                #     raise Exception(f'A node containing text already exists at {found_node.full_node_name}')
                current_parent = found_node
            # If this is the last node and it does not have a heading text, assign it
            if i == len(node_names) - 1 and not current_parent.heading_text:
                current_parent.heading_text = heading_text

    def get_node(self, node_str):
        if node_str == self.root.name:
            return self.root
        if not self.valid_index_checker.is_valid_reference(node_str):
            raise ValueError(f'{node_str} is not a valid node reference')
        # Start search from the root
        current_node = self.root
        node_names = self.valid_index_checker.split_reference(node_str)
        for node_name in node_names:
            # Look for the node among the children of the current node
            found_node = next((node for node in current_node.children if node.name == node_name), None)
            # If not found, raise a ValueError
            if found_node is None:
                raise ValueError(f"Node with path {node_str} does not exist in the tree")
            # If found, continue searching from this node
            current_node = found_node
        # Return the node we've found
        return current_node

    def print_tree(self):
        for pre, _, node in RenderTree(self.root, style=AsciiStyle()):
            print(f"{pre}{node.name} [{node.heading_text}]")

    # I use this function when extracting the headings from the manual for indexing. There are no tests for it yet!!
    # TODO: Add tests for this
    def _list_node_children(self, node, indent = 0):
        string = ""
        # For each node, check if at least one child has a non-empty heading text
        children_with_text = [child for child in node.children if child.heading_text != '']

        if children_with_text:
            # If any child has non-empty heading text, print all that node's children with their heading text
            for child in node.children:
                if child.parent == self.root:
                    if child.name in self.valid_index_checker.exclusion_list:
                        string = string + (' ' * indent + f'{child.name}\n')    
                    else:
                        string = string + (' ' * indent + f'{child.name} {child.heading_text}\n')
                else:
                    string = string + (' ' * indent + f'{child.name} {child.heading_text}\n')
                string = string + self._list_node_children(child, indent + 4)
        return string


def build_tree_for_regulation(root_node_name, regs_as_dataframe, valid_index_checker):
    # Create a tree from the full_reference column and check there are no errors
    tree = Tree(root_node_name, valid_index_checker=valid_index_checker)
    # NOTE: Hierarchy: Regulation / Sub regulation / Paragraph / 
    for i, row in regs_as_dataframe.iterrows() :
        try:
            heading_text = ''
            if row['Heading'] == True:
                heading_text = row['Text']
            if not valid_index_checker.is_valid_reference(row['full_reference']):
                raise ValueError(row['full_reference'] + ' is not a valid reference. See row ' + str(i))
            tree.add_to_tree(row['full_reference'], heading_text=heading_text)
        except Exception as e:
            print(f"An error occurred at row {i}:")
            print(regs_as_dataframe.iloc[i])
            print(f"Error message: {e}")
            break
    return tree


def _split_recursive(node, df, token_limit, valid_index_checker, node_list=[]):
    # Get the full text for this node
    subsection_text = get_regulation_detail(node.full_node_name, df, valid_index_tracker=valid_index_checker)
    token_count = num_tokens_from_string(subsection_text)

    # Check if the total token count of the node is greater than the limit
    if token_count > token_limit:
        # If the token count is over the limit, recursively apply the function to each child
        if len(node.children) == 0:
            raise Exception(f'Node {node.full_node_name} has no children but has a token count of {token_count}')
        for child in node.children:
            _split_recursive(child, df, token_limit, valid_index_checker, node_list)
    else:
        # If the token count is under the limit, add this node to the list
        node_list.append(node)
    return node_list


####
# Starting at an particular parent node (can be the tree root or any child), this method splits up the branch 
# into sections where the text does not exceed a certain token_count cap.
#
# Initially this is used to set up the base DataFrame using node == root and later it can be used if we want 
# to change the word_limit for a specific piece of regulation
###
def split_tree(node, df, token_limit, valid_index_checker):
    node_list=[]
    node_list = _split_recursive(node, df, token_limit, valid_index_checker, node_list)
    section_token_count = []
    for node in node_list:
        #subsection_text = get_full_text_for_node(node.full_node_name, df, False)
        subsection_text = get_regulation_detail(node.full_node_name, df, valid_index_checker)
        token_count = num_tokens_from_string(subsection_text)
        section_token_count.append([node.full_node_name, subsection_text, token_count])

    column_names = ['section', 'text', 'token_count']
    return pd.DataFrame(section_token_count, columns=column_names)


