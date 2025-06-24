# Connect using mongosh
mongosh "mongodb://musequill:musequill.ink.user@localhost:27117/musequill?authSource=musequill"

mongosh "mongodb://musequill:musequill.ink.user@localhost:27017/musequill?authSource=musequill"

# Browse the books collection
use musequill
db.books.find().pretty()


# Find book by _id
use musequill

db.books.find({ _id: "0345cb14-177a-463a-8194-9632bb34c417" }).pretty()
