################################################################
#
# python topwords.py > top_words.txt
#
# Convert LDA topic-words distribution to top N words per topic on stdout
# and generate topic clouds as theme_clouds.pdf in current directory
#
# Input: hard-wires word_topics.csv as input
#   Columns: Word, Pr(word|topic1), ..., Pr(word|topicN)
#
# Output: file that can be read as text or opened as .tsv
#   label <tab> top N words
#
# Also produces theme_label.pdf for each theme
#
################################################################
import os
import sys
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt


# Hardwire how many top words to show for each topic
N = 30
# Hardwire how many top words to show in each cloud
cloudN = 100


# Generate topic cloud PDF files, one per topic
def generate_topic_cloud_frequencies(sorted_df, label, cloudN):
    
    # Get top cloudN rows by value for the relevant column (label)
    # Note: nlargest would work just as well with unsorted dataframe df
    df_top_rows = sorted_df.nlargest(cloudN,label)
    
    # Convert to a dictionary and return
    freqs = dict(zip(df_top_rows['Word'],df_top_rows[label]))
    return(freqs)

# Argument: dict with word:probability (or word:frequency) pairs    
# https://github.com/amueller/word_cloud/blob/master/wordcloud/wordcloud.py#L150
#  To create a word cloud with a single color, use
#  ``color_func=lambda *args, **kwargs: "white"``.
#  The single color can also be specified using RGB code. For example
#  ``color_func=lambda *args, **kwargs: (255,0,0)`` sets color to red.
# Use plt.show() rather than plt.savefig to display figure on screen
# Use "foo.png" instead of "foo.pdf" to generate PNG file
def generate_topic_cloud_image(freq_pairs, outfile_name):
    wordcloud = WordCloud(max_words=2000,
                          background_color="White",
                          prefer_horizontal=1,
                          relative_scaling=0.75,
                          max_font_size=60,
                          random_state=13)
    wordcloud.generate_from_frequencies(freq_pairs)
    # color_func=lambda *args, **kwargs:(100,100,100) # gray
    color_func=lambda *args, **kwargs:(0,126,157) 
    wordcloud.recolor(color_func=color_func, random_state=3)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(outfile_name,bbox_inches='tight')

def write_pdf_with_title(titlestring, pdf_in, pdf_out):
    # https://stackoverflow.com/questions/1180115/add-text-to-existing-pdf-using-python
    # https://stackoverflow.com/questions/9855445/how-to-change-text-font-color-in-reportlab-pdfgen
    from PyPDF2 import PdfFileWriter, PdfFileReader
    import io
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter

    packet = io.BytesIO()
    # create a new PDF with Reportlab
    can = canvas.Canvas(packet, pagesize=letter)
    can.setFillColorRGB(0,0,1) #choose your font colour
    can.setFont("Helvetica", 20) #choose your font type and font size
    can.drawString(10, 10, titlestring)
    can.save()
    #move to the beginning of the StringIO buffer
    packet.seek(0)
    new_pdf = PdfFileReader(packet)
    # read your existing PDF
    existing_pdf = PdfFileReader(open(pdf_in, "rb"))
    output = PdfFileWriter()
    # add the "watermark" (which is the new pdf) on the existing page
    page = existing_pdf.getPage(0)
    page.mergePage(new_pdf.getPage(0))
    output.addPage(page)
    # finally, write "output" to a real file
    outputStream = open(pdf_out, "wb")
    output.write(outputStream)
    outputStream.close()
    
def combine_pdfs(pdfs, outfile_name):
    # Solution to merging PDFs into a multi-page PDF document
    # https://stackoverflow.com/questions/3444645/merge-pdf-files
    from PyPDF2 import PdfFileMerger
    merger = PdfFileMerger()
    for pdf in pdfs:
        merger.append(pdf)
    merger.write(outfile_name)
    merger.close()

def combine_pdf_ALTERNATIVE(pdfs, outfile_name):
    # Another solution to merging PDFs into a multi-page PDF document
    # https://kittaiwong.wordpress.com/2020/09/03/how-to-merge-pdf-files-with-python-pypdf2/
    from PyPDF2 import PdfFileReader, PdfFileWriter
    pdf_writer = PdfFileWriter()
    for path in pdfs:
        pdf_reader = PdfFileReader(path, strict=False)
        for page in range(pdf_reader.getNumPages()):
            pdf_writer.addPage(pdf_reader.getPage(page))
            if page == 0:
                pdf_writer.addBookmark(os.path.basename(path), pdf_writer.getNumPages()-1, parent=None)
    resultPdf = open(outfile_name, 'wb')
    pdf_writer.write(resultPdf)
    resultPdf.close()
    
################################################################
# Main
################################################################    

# Read CSV file into DataFrame df
df = pd.read_csv('word_topics.csv')

# Get theme labels
labels = list(df.columns.values)
pdfs   = []

# Generate topic cloud PDF file for each theme
for label in labels:

    # Filenames
    # TO DO: make these temporary files that get cleaned up
    image_file_name  = label.replace(" ", "_") + ".pdf"
    temp_file_name   = "TEMP_" + image_file_name

    # For each theme label, sort descending and report top N words to stdout
    if label == 'Word':
        continue
    sorted_df = df.sort_values(by = label, ascending = False)
    words    = sorted_df['Word'].tolist()
    topwords = words[:N]
    print(label + "\t" + " ".join(topwords))

    # Generate image to temporary PDF file.
    # Skip topic if generate an image for it has problems. (Debug in future!)
    freqs           = generate_topic_cloud_frequencies(sorted_df, label, cloudN)
    try:
        generate_topic_cloud_image(freqs, temp_file_name)
        # Add label at top to create image PDF file
        sys.stderr.write("Writing " + temp_file_name + "\n")
        write_pdf_with_title(label, temp_file_name, image_file_name)
        pdfs.append(image_file_name)
    except:
        sys.stderr.write("Unable to create cloud image for " + label + ". Skipping.\n")


# Combine into one PDF
combine_pdfs(pdfs, 'theme_clouds.pdf')

sys.stderr.write("Theme clouds are in theme_clouds.pdf\n")
sys.stderr.write("To clean up: rm TEMP_*.pdf\n")



