import sqlite3

## connect to sqllite
connection=sqlite3.connect("users.db")

##create a cursor object to insert record,create table
cursor=connection.cursor()


try:
    
    ## create the table
    table_info="""
        create table UserManagement(FIRST_NAME VARCHAR(25),
                                   LAST_NAME VARCHAR(25),
                                   password VARCHAR(25),
                                   LOCATION VARCHAR(25),
                                   phone_number VARCHAR(25),
                                   email VARCHAR(255)
                                   MARKS INT)
        """


    cursor.execute(table_info)

    insert_commands = """
        INSERT INTO UserManagement (FIRST_NAME, LAST_NAME, password, LOCATION, phone_number, email, MARKS)
        VALUES
        ('John', 'Doe', 'password123', 'Visakhapanam', '123-456-7890', 'john.doe@example.com', 85),
        ('Jane', 'Smith', 'password456', 'Delhi', '987-654-3210', 'jane.smith@example.com', 92),
        ('Alice', 'Johnson', 'password789', 'HYD', '555-123-4567', 'alice.johnson@example.com', 78),
        ('Bob', 'Williams', 'password101', 'Chennai', '555-987-6543', 'bob.williams@example.com', 88),
        ('Charlie', 'Brown', 'password202', 'HYD', '555-555-5555', 'charlie.brown@example.com', 91)
    """
    cursor.execute(insert_commands)


except Exception as e:
    print('Exception',e)
    ## Commit your changes in the database
    
    ## Display all the records
#print("The inserted records are")
data=cursor.execute('''Select * from UserManagement''')
for row in data:
    print(row)
connection.commit()
connection.close()
