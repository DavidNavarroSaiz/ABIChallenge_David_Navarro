
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Retrieve the database URL from environment variables
DATABASE_URL = os.getenv("DATABASE_URL")

# Create a SQLAlchemy engine for database interaction
engine = create_engine(DATABASE_URL)

# Create a base class for database models using the SQLAlchemy declarative_base() function
Base = declarative_base()
