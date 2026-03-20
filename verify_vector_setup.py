#!/usr/bin/env python3
"""
向量数据库集成验证脚本
检查所有组件是否正确配置和工作
"""

import sys
import os
from pathlib import Path

# 添加项目目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def print_section(title):
    """打印分节标题"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def print_success(msg):
    """打印成功消息"""
    print(f"✓ {msg}")


def print_error(msg):
    """打印错误消息"""
    print(f"✗ {msg}")


def print_info(msg):
    """打印信息消息"""
    print(f"ℹ {msg}")


def step1_check_dependencies():
    """步骤1：检查依赖包"""
    print_section("步骤 1: 检查依赖包")
    
    packages = [
        ("psycopg2", "PostgreSQL 驱动"),
        ("pgvector", "向量扩展"),
        ("sentence_transformers", "嵌入模型"),
        ("langchain", "LangChain 框架"),
    ]
    
    all_ok = True
    for pkg_name, description in packages:
        try:
            __import__(pkg_name.replace("-", "_"))
            print_success(f"{description} ({pkg_name})")
        except ImportError:
            print_error(f"{description} ({pkg_name}) - 未安装")
            all_ok = False
    
    return all_ok


def step2_check_config_loading():
    """步骤2：检查配置加载"""
    print_section("步骤 2: 检查配置加载")
    
    try:
        from config.get_config import config_data
        
        vector_config = config_data.get("vector", {})
        if not vector_config:
            print_error("配置文件中没有 vector 部分")
            return False
        
        print_success("配置加载成功")
        print_info(f"  enabled: {vector_config.get('enabled')}")
        print_info(f"  embedding_model: {vector_config.get('embedding_model')}")
        print_info(f"  embedding_device: {vector_config.get('embedding_device')}")
        print_info(f"  host: {vector_config.get('db', {}).get('host')}")
        print_info(f"  port: {vector_config.get('db', {}).get('port')}")
        print_info(f"  database: {vector_config.get('db', {}).get('database')}")
        print_info(f"  user: {vector_config.get('db', {}).get('user')}")
        print_info(f"  collection_name: {vector_config.get('collection_name')}")
        print_info(f"  top_k: {vector_config.get('top_k')}")
        print_info(f"  distance_strategy: {vector_config.get('distance_strategy')}")
        
        return True
    except Exception as e:
        print_error(f"配置加载失败: {e}")
        return False


def step3_check_postgresql_connection():
    """步骤3：检查 PostgreSQL 连接"""
    print_section("步骤 3: 检查 PostgreSQL 连接")
    
    try:
        import psycopg2
        from config.get_config import config_data
        
        db_config = config_data["vector"]["db"]
        conn = psycopg2.connect(
            host=db_config["host"],
            port=db_config["port"],
            database=db_config["database"],
            user=db_config["user"],
            password=db_config["password"]
        )
        
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.close()
        
        print_success(f"成功连接到 PostgreSQL: {db_config['user']}@{db_config['host']}:{db_config['port']}/{db_config['database']}")
        
        # 检查 pgvector 扩展
        cursor = conn.cursor()
        cursor.execute("SELECT extname FROM pg_extension WHERE extname = 'vector'")
        result = cursor.fetchone()
        cursor.close()
        
        if result:
            print_success("pgvector 扩展已安装")
        else:
            print_error("pgvector 扩展未安装")
            conn.close()
            return False
        
        # 检查表
        cursor = conn.cursor()
        cursor.execute("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables 
                WHERE table_name = 'langchain_pg_collection'
            )
        """)
        result = cursor.fetchone()[0]
        cursor.close()
        
        if result:
            print_success("表结构已初始化")
        else:
            print_error("表结构未初始化 - 需要运行 pgv/create_table.sql")
            conn.close()
            return False
        
        conn.close()
        return True
        
    except Exception as e:
        print_error(f"PostgreSQL 连接失败: {e}")
        return False


def step4_check_embedding_model():
    """步骤4：检查嵌入模型"""
    print_section("步骤 4: 检查嵌入模型")
    
    try:
        from config.get_config import config_data
        from pgv.embedding import get_embedding_function
        
        vector_config = config_data["vector"]
        model_name = vector_config["embedding_model"]
        device = vector_config["embedding_device"]
        
        print_info(f"加载嵌入模型: {model_name}")
        print_info(f"设备: {device}")
        print_info("(首次运行可能需要下载~400MB 的模型，请耐心等待...)")
        
        embedding_func = get_embedding_function(vector_config, project_root)
        
        # 测试嵌入
        test_texts = ["test"]
        embeddings = embedding_func.embed_documents(test_texts)
        
        print_success(f"嵌入模型已加载")
        print_info(f"向量维度: {len(embeddings[0])}")
        
        return True
        
    except Exception as e:
        print_error(f"嵌入模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def step5_check_vector_service():
    """步骤5：检查向量服务"""
    print_section("步骤 5: 检查向量服务")
    
    try:
        from pgv.ask import get_schema_vector_service
        
        service = get_schema_vector_service()
        
        if service is None:
            print_error("向量服务未初始化")
            return False
        
        is_enabled = service.is_enabled()
        
        if is_enabled:
            print_success("向量服务已启用")
        else:
            print_error("向量服务已禁用")
            return False
        
        return True
        
    except Exception as e:
        print_error(f"向量服务检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def step6_check_schema_integration():
    """步骤6：检查与主系统的集成"""
    print_section("步骤 6: 检查与主系统的集成")
    
    try:
        # 检查导入
        from ask_ai.ask_api import get_final_prompt
        from pgv.ask import build_semantic_context
        
        print_success("主系统导入成功")
        
        # 检查配置是否一致
        from config.get_config import config_data
        mysql_config = config_data.get("mysql")
        vector_config = config_data.get("vector")
        
        if mysql_config:
            print_success("MySQL 配置已加载")
        else:
            print_error("MySQL 配置缺失")
            return False
        
        if vector_config and vector_config.get("enabled"):
            print_success("向量模块已启用")
        else:
            print_info("向量模块已禁用 (可在 config.yaml 中启用)")
        
        return True
        
    except Exception as e:
        print_error(f"集成检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def step7_test_semantic_search():
    """步骤7：测试语义搜索"""
    print_section("步骤 7: 测试语义搜索（模拟）")
    
    try:
        from config.get_config import config_data
        
        if not config_data["vector"].get("enabled"):
            print_info("向量模块未启用，跳过此步骤")
            return True
        
        print_info("此步骤需要有效的表结构数据")
        print_info("完整测试建议在启动主服务后进行:")
        print_info("  1. 启动服务: python main.py")
        print_info("  2. 调用 API: curl http://localhost:8087/ask/pd -X POST")
        print_info("  3. 检查日志中是否有 'Synced X schema semantic documents'")
        
        return True
        
    except Exception as e:
        print_error(f"测试失败: {e}")
        return False


def main():
    """主验证流程"""
    print("\n" + "="*60)
    print("  向量数据库系统验证")
    print("="*60)
    
    steps = [
        ("依赖包检查", step1_check_dependencies),
        ("配置加载", step2_check_config_loading),
        ("PostgreSQL 连接", step3_check_postgresql_connection),
        ("嵌入模型", step4_check_embedding_model),
        ("向量服务", step5_check_vector_service),
        ("主系统集成", step6_check_schema_integration),
        ("语义搜索", step7_test_semantic_search),
    ]
    
    results = []
    for step_name, step_func in steps:
        try:
            result = step_func()
            results.append((step_name, result))
        except Exception as e:
            print_error(f"步骤执行异常: {e}")
            import traceback
            traceback.print_exc()
            results.append((step_name, False))
    
    # 总结
    print_section("验证总结")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for step_name, result in results:
        status = "✓" if result else "✗"
        print(f"{status} {step_name}")
    
    print(f"\n总计: {passed}/{total} 项检查通过")
    
    if passed == total:
        print_success("所有检查都通过了！系统已准备就绪。")
        print_info("下一步:")
        print_info("  1. 运行 python main.py 启动服务")
        print_info("  2. 首次启动会自动索引表结构到向量数据库")
        print_info("  3. 调用 API 进行测试")
        return 0
    else:
        print_error(f"有 {total - passed} 项检查失败，请查看上面的错误信息。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
