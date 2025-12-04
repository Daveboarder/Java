
import sqlite3
import csv


# Connect to database (creates file if it doesn't exist)
conn = sqlite3.connect('QuantParam.db')
# Create a cursor to execute SQL commands
cursor = conn.cursor()

# Example: Create a table
# id INTEGER PRIMARY KEY automatically assigns sequential numbers (1, 2, 3, ...)
# AUTOINCREMENT ensures it always increments even after deletions (optional)
cursor.execute('''
CREATE TABLE IF NOT EXISTS QuantParam (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Elem_name TEXT,
    ion_state TEXT,
    Wavelength REAL,
    Ei REAL,
    Ek REAL,
    Ak REAL,
    gi REAL,
    gk REAL
)''')

# Insert data
# Note: id is not included in INSERT - SQLite will automatically assign the next number
with open('Quant_par.txt', 'r') as file:
    reader = csv.DictReader(file, delimiter='\t')
    for row in reader:
        cursor.execute('INSERT INTO QuantParam (Elem_name, ion_state, Wavelength, Ei, Ek, Ak, gi, gk) VALUES (?, ?, ?, ?, ?, ?, ?, ?)', 
        (row['Elem_name'], row['ion.state'], row['Wl'], row['Ei'], row['Ek'], row['Ak'], row['gi'], row['gk']))

# Commit changes and close
conn.commit()
conn.close()
