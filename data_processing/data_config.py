import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
import pandas as pd

# 加载环境变量
load_dotenv()

class ConfigData:
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir
        
        # 修改为15分钟数据的CSV文件路径
        self.csv_path = f"{data_dir}/BTC_15m.csv"
        
        # 修改为15分钟数据的文件路径
        self.input_ary_path = f"{data_dir}/BTC_15m_input.npy"
        self.label_ary_path = f"{data_dir}/BTC_15m_label.npy"
        self.predict_ary_path = f"{data_dir}/BTC_15m_input.npy"  # 用于预测的输入数据
        self.predict_net_path = f"{data_dir}/BTC_15m_predict.pth"
        
        # 数据库配置
        self.db_host = os.getenv('DB_HOST')
        self.db_port = os.getenv('DB_PORT')
        self.db_name = os.getenv('DB_NAME')
        self.db_user = os.getenv('DB_USER')
        self.db_password = os.getenv('DB_PASSWORD')
        self.btc_table = os.getenv('BTCSPOT_TABLE')
        
    def get_db_connection_string(self):
        """获取数据库连接字符串"""
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
    
    def load_btc_data_from_db(self, limit=None):
        """从PostgreSQL数据库加载BTC 15分钟K线数据"""
        try:
            # 创建数据库连接
            engine = create_engine(self.get_db_connection_string())
            
            # 构建SQL查询，根据实际数据表结构
            sql_query = f"""
            SELECT 
                datetime as system_time,
                (high + low) / 2 as midpoint,
                high - low as spread,
                volume as buys,
                volume as sells,
                (high - low) * 0.1 as bids_distance_0, (high - low) * 0.1 as asks_distance_0,
                volume as bids_notional_0, volume as asks_notional_0,
                0 as bids_cancel_notional_0, 0 as asks_cancel_notional_0,
                volume as bids_limit_notional_0, volume as asks_limit_notional_0,
                (high - low) * 0.2 as bids_distance_1, (high - low) * 0.2 as asks_distance_1,
                volume * 0.8 as bids_notional_1, volume * 0.8 as asks_notional_1,
                0 as bids_cancel_notional_1, 0 as asks_cancel_notional_1,
                volume * 0.8 as bids_limit_notional_1, volume * 0.8 as asks_limit_notional_1,
                (high - low) * 0.3 as bids_distance_2, (high - low) * 0.3 as asks_distance_2,
                volume * 0.6 as bids_notional_2, volume * 0.6 as asks_notional_2,
                0 as bids_cancel_notional_2, 0 as asks_cancel_notional_2,
                volume * 0.6 as bids_limit_notional_2, volume * 0.6 as asks_limit_notional_2,
                (high - low) * 0.4 as bids_distance_3, (high - low) * 0.4 as asks_distance_3,
                volume * 0.4 as bids_notional_3, volume * 0.4 as asks_notional_3,
                0 as bids_cancel_notional_3, 0 as asks_cancel_notional_3,
                volume * 0.4 as bids_limit_notional_3, volume * 0.4 as asks_limit_notional_3,
                (high - low) * 0.5 as bids_distance_4, (high - low) * 0.5 as asks_distance_4,
                volume * 0.2 as bids_notional_4, volume * 0.2 as asks_notional_4,
                0 as bids_cancel_notional_4, 0 as asks_cancel_notional_4,
                volume * 0.2 as bids_limit_notional_4, volume * 0.2 as asks_limit_notional_4
            FROM {self.btc_table}
            WHERE valid = true
            ORDER BY datetime
            """
            
            if limit:
                sql_query += f" LIMIT {limit}"
            
            # 执行查询并返回DataFrame
            df = pd.read_sql(sql_query, engine)
            print(f"成功从数据库加载 {len(df)} 条15分钟K线数据")
            return df
            
        except Exception as e:
            print(f"从数据库加载数据失败: {e}")
            return None