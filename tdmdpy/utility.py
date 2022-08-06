import fileinput
import sys


def delete_specific_line(file_name, key_words):
    """delete specific line from one text file
           input:
           file_name: (str) name of the file

           key_words: (str) pattern key words that needed to be deleted
    """
    a_file = open(file_name, "r")

    lines = a_file.readlines()
    a_file.close(),

    new_file = open(file_name, "w")
    for line in lines:
        if line.find(key_words) == -1:
            new_file.write(line)

    new_file.close()


def extract_sections_from_txt(file_name,
                              spacing=1,
                              section_index=1,
                              section_txt_name=None):

    """ extract information from part of the txt file
           input:
           file_name: (str) Name of the file
           spacing: (int) Integer spacing of file
           section_index: (int) Index of section in a txt file
           section_txt_name: (str) Name of the section txt

    """
    # First pass: Get all data with read_lines()
    file_object = open(file_name, 'r')
    contents = file_object.readlines()

    # Derive start and end index from section index
    start_index = (section_index - 1) * spacing
    end_index = start_index + spacing

    # Write section out
    if section_txt_name is None:
        section_txt_name = 'tmp_step_' + str(section_index) + '.txt'
    new_file_object = open(section_txt_name, 'r')
    for i in range(start_index, end_index + 1):
        new_file_object.write(contents[i])
            

def replace_line(file_name, line_num, text):
    """replace specific content in one text file based on line number
       input:
       file_name: (str) name of the file
       line_num: (int) line number
       text: (str) content to replace

    """
    lines = open(file_name, 'r').readlines()
    lines[line_num] = text
    out = open(file_name, 'w')
    out.writelines(lines)
    out.close()


def replaceAll(file_name, searchExp, replaceExp):
    """replace specific content in one text file
              input:
              file_name: (str) name of the file

              searchExp: (str) pattern expression to be replaced in the text file

              replaceExp: (str) pattern expression replace into the text file
    """
    for line in fileinput.input(file_name, inplace=1):
        if searchExp in line:
            line = line.replace(searchExp, replaceExp)
        sys.stdout.write(line)
