import doctable
import dataclasses
import pathlib
import sqlalchemy

class DistanceDB:
    core: doctable.ConnectCore
    tab: doctable.DBTable
    
    @classmethod
    def open(cls, dbpath: pathlib.Path) -> DistanceDB:
        core = doctable.ConnectCore.open(dbpath, dialect='sqlite')
        with core.begin_ddl() as ddl:
            tab = ddl.create_table_if_not_exists(Distance)
        return cls(core=core, tab=tab)

class Query:
    db: DistanceDB
    
    top_distances_sql: str = '''WITH cte AS (
            SELECT target_path, position, thumb, distance, 
            RANK() OVER ( 
                PARTITION BY target_path,position 
                ORDER BY distance ASC
            ) AS r FROM Distance 
        ) 
        SELECT target_path, position, thumb, distance FROM cte 
        WHERE r <= {n} 
        ORDER BY distance ASC;'
    '''
    def select_top_distances(n: int) -> sqlalchemy.RowProxy:
        return self.db.core.execute_sql(self.top_distances_sql.format(n=n))
    

@doctable.table_schema(
    indices = {
        'target_position_thumb': doctable.Index('target_path', 'position', 'thumb', unique=True),
    },
)
class Distance:
    target_path: str
    position: int
    thumb: str
    distance: float
    
    id: int = doctable.Column(
        column_args=doctable.ColumnArgs(
            order=0,
            primary_key=True,
            autoincrement=True,
        )
    )


