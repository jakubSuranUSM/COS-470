import requests
from bs4 import BeautifulSoup
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
from transformers import AutoTokenizer
from nltk import word_tokenize
import nltk
nltk.download('punkt')

godFather_link = "https://www.dailyscript.com/scripts/The_Godfather.html"
godFather2_link = "https://www.scripts.com/script.php?id=the_godfather_71&p="


# -------------------------------- TASK 1 --------------------------------
def extract_character_lines1():
    """
       Extracts lines of dialogue for the character "MICHAEL" from the first web page containing the script of "The Godfather".

       Returns:
       List of strings: Each string represents a line of dialogue spoken by the character "MICHAEL".
    """
    # download HTML
    html_gf1 = requests.get(godFather_link)
    bs_gf1 = BeautifulSoup(html_gf1.text, "html.parser")

    # find block with the content
    text = bs_gf1.find('pre').text

    result = []
    current_line = None

    # extract MICHAEL's lines
    for line in text.splitlines():
        if re.search('\t\t\t\tMICHAEL', line):
            if current_line is not None:
                result.append(current_line)
            current_line = ''
        elif current_line is not None:
            current_line += line
            if not line.strip():
                current_line = current_line.replace('\n', ' ')
                current_line = current_line.replace('\t', ' ')
                current_line = re.sub('(\\(.*?\\)|^ \\s*\\(.*?\\))', ' ', current_line)
                current_line = re.sub('\\s{2,}', ' ', current_line)
                result.append(current_line.strip())
                current_line = None
    return result


def extract_character_lines2():
    """
       Extracts lines of dialogue for the character "MICHAEL" from the second web page containing the script of "The Godfather".

       Returns:
       List of strings: Each string represents a line of dialogue spoken by the character "MICHAEL".
    """
    result = []

    for i in range(1, 103):
        # download HTML
        html = requests.get(godFather2_link + str(i))

        # find content area
        text = re.findall('<blockquote>.*</blockquote>', html.text, re.DOTALL)[0]

        # extract MICHAEL's lines
        character_blocks = re.findall('MICHAEL:</strong>.*?<strong>|MICHAEL:</strong>.*?</div>', text, re.DOTALL)
        for block in character_blocks:
            line = re.sub(r'<p>|<a[^>]*>|</a>|\(.*?\)|MICHAEL:</strong>|<strong>|</div>', '', block)
            line = re.sub('</p>', ' ', line)
            line = re.sub('\\s{2,}', ' ', line)
            pattern = re.compile(r'([A-Z]{3,})')
            match = pattern.search(line)
            if match:
                line = line[:match.start()]
            result.append(line.strip())
    return result


def scrape():
    """
        Scrapes character lines from two different sources related to the Godfather movie, extracts the lines, and saves them to text files.

        Returns:
        None
    """
    extracted_lines1 = extract_character_lines1()
    gf1 = '\n'.join(extracted_lines1)
    f = open("GodFather1.txt", "w")
    f.write(gf1)
    f.close()

    extracted_lines2 = extract_character_lines2()
    gf2 = '\n'.join(extracted_lines2)
    f = open("GodFather2.txt", "w")
    f.write(gf2)
    f.close()


def get_scripts():
    """
        Retrieves the text content of two Godfather movie scripts from files, scraping them if the files do not exist.

        Returns:
        tuple: A tuple containing the text content of the first and second Godfather movie scripts.
    """
    if not all(os.path.exists(file) for file in ['GodFather1.txt', 'GodFather2.txt']):
        scrape()
    f = open('GodFather1.txt', 'r')
    gf1 = f.read()
    f.close()

    f = open('GodFather2.txt', 'r')
    gf2 = f.read()
    f.close()
    return gf1, gf2


# -------------------------------- TASK 2 --------------------------------
def get_word_clouds(gf1, gf2):
    """
        Generates and displays word clouds for two Godfather movie scripts.

        Parameters:
        - gf1 (str): Text content of the first Godfather movie script.
        - gf2 (str): Text content of the second Godfather movie script.

        Returns:
        None
    """
    wordcloud1 = WordCloud(width=800, height=400,
                           background_color='white').generate(gf1)
    wordcloud2 = WordCloud(width=800, height=400,
                           background_color='white').generate(gf2)

    plt.figure(figsize=(16, 8))

    plt.subplot(1, 2, 1)
    plt.imshow(wordcloud1)
    plt.axis('off')
    plt.title('The Godfather 1', fontsize=20, y=1.1)

    plt.subplot(1, 2, 2)
    plt.imshow(wordcloud2)
    plt.axis('off')
    plt.title('The Godfather 2', fontsize=20, y=1.1)

    plt.savefig('plots.pdf', bbox_inches='tight')
    plt.show()


# -------------------------------- TASK 3 --------------------------------
def tokenize():
    """
        Demonstrates tokenization using both WordPiece tokenization from BERT and word tokenization from NLTK.

        Returns:
        None
    """
    line = 'Because they know that no Sicilian will refuse a request on his daughter\'s wedding day.'
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    encoding = tokenizer.encode(line)
    print("WordPiece tokenization")
    print(tokenizer.convert_ids_to_tokens(encoding))

    tokens = word_tokenize(line)
    print("Word tokenization")
    print(tokens)


def main():
    # TASK 1
    gf1, gf2 = get_scripts()

    # TASK 2
    get_word_clouds(gf1, gf2)

    # TASK 3
    tokenize()

    return


if __name__ == "__main__":
    main()
