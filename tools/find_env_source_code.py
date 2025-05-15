import gymnasium as gym

env_id = "MountainCar-v0"  # 替换为你感兴趣的环境 ID
# env_id = "MountainCar-v0"
# env_id = "PongNoFrameskip-v4" # 对于 Atari 环境，这会指向 ALE 包装器

try:
    env = gym.make(env_id)
    if env.spec:
        print(f"环境 ID: {env_id}")
        print(f"入口点 (Entry Point): {env.spec.entry_point}")

        # 从入口点推断模块和类/函数名
        # 格式通常是 'module.path:ClassName' 或 'module.path:function_name'
        module_path, object_name = env.spec.entry_point.split(':')
        print(f"模块路径: {module_path}")
        print(f"对象名称 (类/函数): {object_name}")

        # 尝试导入模块并获取对象
        try:
            module = __import__(module_path, fromlist=[object_name])
            env_object = getattr(module, object_name)
            import inspect
            source_file = inspect.getfile(env_object)
            print(f"源代码文件路径: {source_file}")
        except ImportError:
            print(f"无法导入模块: {module_path}")
        except AttributeError:
            print(f"在模块 {module_path} 中找不到对象: {object_name}")
        except TypeError:
            # 如果 env_object 是一个内置类型或C扩展，getfile 可能失败
            print(f"无法获取 {object_name} 的源文件 (可能是内置或C扩展)")

    else:
        print(f"环境 {env_id} 没有 spec 属性。")
    env.close()

except gym.error.Error as e:
    print(f"创建环境 {env_id} 时出错: {e}")
except Exception as e:
    print(f"发生未知错误: {e}")