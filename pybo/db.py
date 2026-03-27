import oracledb
from sqlalchemy import create_engine
from config import SQLALCHEMY_DATABASE_URI

# Oracle Instant Client
oracledb.init_oracle_client(lib_dir=r"C:\oraclexe\instantclient_19_25")

engine = create_engine(SQLALCHEMY_DATABASE_URI)