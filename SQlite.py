
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
        #check if the row already exists
        cursor.execute("SELECT * FROM QuantParam WHERE Elem_name = ? AND ion_state = ? AND Wavelength = ?", (row['Elem_name'], row['ion_state'], row['Wl']))
        if cursor.fetchone() is None:
            cursor.execute('INSERT INTO QuantParam (Elem_name, ion_state, Wavelength, Ei, Ek, Ak, gi, gk) VALUES (?, ?, ?, ?, ?, ?, ?, ?)', 
            (row['Elem_name'], row['ion_state'], row['Wl'], row['Ei'], row['Ek'], row['Ak'], row['gi'], row['gk']))
        else:
            #print(f"Row already exists: {row['Elem_name']}, {row['ion_state']}, {row['Wl']}")
            cursor.execute("UPDATE QuantParam SET Ei = ?, Ek = ?, Ak = ?, gi = ?, gk = ? WHERE Elem_name = ? AND ion_state = ? AND Wavelength = ?", (row['Ei'], row['Ek'], row['Ak'], row['gi'], row['gk'], row['Elem_name'], row['ion_state'], row['Wl']))

# Commit changes and close
conn.commit()
conn.close()

conn = sqlite3.connect('PartF_var.db')
# Create a cursor to execute SQL commands
cursor = conn.cursor()

# Example: Create a table
# id INTEGER PRIMARY KEY automatically assigns sequential numbers (1, 2, 3, ...)
# AUTOINCREMENT ensures it always increments even after deletions (optional)
cursor.execute('''
CREATE TABLE IF NOT EXISTS PartF_var (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Elem_name TEXT,
    ion_state TEXT,
    Ei REAL,
    gi REAL
)''')

# Insert data
# Note: id is not included in INSERT - SQLite will automatically assign the next number
with open('PartF_var.txt', 'r') as file:
    reader = csv.DictReader(file, delimiter='\t')
    for row in reader:
        #check if the row already exists
        cursor.execute("SELECT * FROM PartF_var WHERE Elem_name = ? AND ion_state = ? AND Ei = ? AND gi = ?", (row['Element'], row['ionState'], row['Ei'], row['gi']))
        if cursor.fetchone() is None:
            cursor.execute('INSERT INTO PartF_var (Elem_name, ion_state, Ei, gi) VALUES (?, ?, ?, ?)', 
            (row['Element'], row['ionState'], row['Ei'], row['gi']))
        else:
            #print(f"Row already exists: {row['Elem_name']}, {row['ion_state']}, {row['Wl']}")
            cursor.execute("UPDATE PartF_var SET Ei = ?, gi = ? WHERE Elem_name = ? AND ion_state = ? AND Ei = ? AND gi = ?", (row['Ei'], row['gi'], row['Element'], row['ionState'], row['Ei'], row['gi']))

# Commit changes and close
conn.commit()
conn.close()

conn = sqlite3.connect('E_ion.db')
# Create a cursor to execute SQL commands
cursor = conn.cursor()

# Example: Create a table
# id INTEGER PRIMARY KEY automatically assigns sequential numbers (1, 2, 3, ...)
# AUTOINCREMENT ensures it always increments even after deletions (optional)
cursor.execute('''
CREATE TABLE IF NOT EXISTS E_ion (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Elem_name TEXT,
    Eion REAL,
    Eion_sd REAL
)''')

# Insert data
# Note: id is not included in INSERT - SQLite will automatically assign the next number
with open('E_ion.txt', 'r') as file:
    reader = csv.DictReader(file, delimiter='\t')
    for row in reader:
        #check if the row already exists
        cursor.execute("SELECT * FROM E_ion WHERE Elem_name = ?", (row['Element'],))
        if cursor.fetchone() is None:
            cursor.execute('INSERT INTO  E_ion (Elem_name, Eion, Eion_sd) VALUES (?, ?, ?)', 
            (row['Element'], row['Eion'], row['Eion_sd']))
        else:
            #print(f"Row already exists: {row['Elem_name']}, {row['ion_state']}, {row['Wl']}")
            cursor.execute("UPDATE E_ion SET Eion = ?, Eion_sd = ? WHERE Elem_name = ?", (row['Eion'], row['Eion_sd'], row['Element']))

# Commit changes and close
conn.commit()
conn.close()
