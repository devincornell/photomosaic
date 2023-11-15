from __future__ import annotations

import typing
import doctable
import dataclasses
import pathlib
import sqlalchemy

@dataclasses.dataclass
class DistanceDB:
    core: doctable.ConnectCore
    tab: doctable.DBTable
    
    @classmethod
    def open(cls, dbpath: pathlib.Path) -> DistanceDB:
        core = doctable.ConnectCore.open(dbpath, dialect='sqlite')
        with core.begin_ddl() as ddl:
            tab = ddl.create_table_if_not_exists(Distance)
        return cls(core=core, tab=tab)
    
    @property
    def q(self):
        return Query(db=self)

@dataclasses.dataclass
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
        ORDER BY distance ASC;
    '''
    def select_top_distances(self, n: int) -> typing.List[Distance]:
        results = self.select_top_distances_sql(n)
        #return results.all()[0]
        return [Distance(**r._mapping) for r in results.all()]
    
    def select_top_distances_sql(self, n: int) -> sqlalchemy.RowProxy:
        core, t = self.core_table()
        return core.query().execute_sql(self.top_distances_sql.format(n=n))
    
    def count_by_target(self) -> int:
        core, t = self.core_table()
        return core.query().select([t['target_path'], doctable.f.count(t['id'])], group_by=[t['target_path']])
    
    def core_table(self) -> doctable.DBTable:
        return self.db.core, self.db.tab
    

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


