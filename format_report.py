import csv
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

print("Formatting data for client deliverable...")

# Create new workbook
wb = Workbook()
ws = wb.active
ws.title = "B2B Market Intelligence"

# Read CSV and write to Excel
try:
    with open('B2B_Financial_Offerings_Extract.csv', mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            ws.append(row)
except FileNotFoundError:
    print("Error: Could not find 'B2B_Financial_Offerings_Extract.csv'.")
    exit(1)

# 1. Style Header Row
header_fill = PatternFill(start_color="1A365D", end_color="1A365D", fill_type="solid")
header_font = Font(bold=True, color="FFFFFF")
for cell in ws[1]:
    cell.fill = header_fill
    cell.font = header_font
    cell.alignment = Alignment(horizontal="center", vertical="center")

# 2. Set Perfect Column Widths
ws.column_dimensions['A'].width = 22  # Date
ws.column_dimensions['B'].width = 30  # URL
ws.column_dimensions['C'].width = 85  # AI Summary (Wide)
ws.column_dimensions['D'].width = 15  # Status

# 3. Apply Text Wrapping and Borders to Data
thin_border = Border(left=Side(style='thin', color='CBD5E1'), 
                     right=Side(style='thin', color='CBD5E1'), 
                     top=Side(style='thin', color='CBD5E1'), 
                     bottom=Side(style='thin', color='CBD5E1'))

for row in ws.iter_rows(min_row=2):
    for cell in row:
        cell.border = thin_border
        cell.alignment = Alignment(vertical="center", wrap_text=True)

# Save the final client-ready file
wb.save('Client_Ready_Deliverable.xlsx')
print("✅ Success! Your file 'Client_Ready_Deliverable.xlsx' is ready to send.")