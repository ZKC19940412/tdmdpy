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

def replaceAll(file, searchExp, replaceExp):
    """replace specific content in one text file
              input:
              file: (str) name of the file

              searchExp: (str) pattern expression to be replaced in the text file

              replaceExp: (str) pattern expression replace into the text file
    """
    for line in fileinput.input(file, inplace=1):
        if searchExp in line:
            line = line.replace(searchExp, replaceExp)
        sys.stdout.write(line)
