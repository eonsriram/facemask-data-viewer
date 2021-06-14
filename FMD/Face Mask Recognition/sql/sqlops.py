import mysql.connector as connector

"""
CREATE TABLE IF NOT EXISTS log(
    id int NOT NULL AUTO_INCREMENT, 
    name varchar(255) NOT NULL, 
    login TIMESTAMP, logout TIMESTAMP, 
    inMask BIT, outMask BIT, PRIMARY KEY (id)
);


# db.read("describe nie.log;", disp=True)
"""


class SQLDB:
    def __init__(self, host="localhost", user="root", password="password", database="nie"):
        self.host = host
        self.user = user
        self.password = password
        self.database_name = database

        self.connection = connector.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            database=self.database_name,
            auth_plugin = 'mysql_native_password'
        )

        self.cursor = self.connection.cursor()

    def getCursor(self):
        return self.cursor

    def write(self, query):
        cur = self.getCursor()
        cur.execute(query)
        self.connection.commit()

    def read(self, query, disp = False):
        cur = self.getCursor()
        cur.execute(query)
        result = cur.fetchall()
        if disp:
            print(result)
        return result

    def insertRecord(self, name, login="2008-01-01 00:00:01", logout="2008-01-01 00:00:01", inmask='0', outmask='0'):
        query = """INSERT INTO nie.log 
        (name, login, logout, inMask, outMask)
         VALUES ('{}','{}','{}',{},{});
         """.format(name, login, logout, inmask, outmask)

        self.write(query)

    def showTable(self):
        query = "SELECT * FROM nie.log LIMIT 100;"

        self.read(query, disp=True)

    def emptyTable(self, table="log"):
        self.write("DELETE FROM {}.{}".format(self.database_name, table))
        self.write("ALTER TABLE nie.log AUTO_INCREMENT = 1")


    def close(self):
        self.connection.close()
        self.cursor.close()


def main():
    return


if __name__ == "__main__":
    main()
