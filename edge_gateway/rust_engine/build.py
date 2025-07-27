"""
EdgeFit Rust Engine 构建脚本
用于编译 Rust 组件并生成 Python 绑定
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path

class RustEngineBuilder:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.rust_dir = self.project_root
        self.target_dir = self.rust_dir / "target"
        self.python_target = self.project_root.parent / "rust_engine"
        
    def check_dependencies(self):
        """检查构建依赖"""
        print("🔍 检查构建依赖...")
        
        # 检查 Rust
        try:
            result = subprocess.run(["rustc", "--version"], capture_output=True, text=True)
            print(f"✅ Rust: {result.stdout.strip()}")
        except FileNotFoundError:
            print("❌ Rust 未安装，请访问 https://rustup.rs/ 安装")
            sys.exit(1)
            
        # 检查 maturin
        try:
            result = subprocess.run(["maturin", "--version"], capture_output=True, text=True)
            print(f"✅ Maturin: {result.stdout.strip()}")
        except FileNotFoundError:
            print("📦 安装 maturin...")
            subprocess.run([sys.executable, "-m", "pip", "install", "maturin"], check=True)
            
        # 检查 PyTorch (for tch crate)
        try:
            import torch
            print(f"✅ PyTorch: {torch.__version__}")
        except ImportError:
            print("⚠️  PyTorch 未安装，tch crate 可能无法正常工作")
            print("   请运行: pip install torch torchvision")
    
    def clean_build(self):
        """清理构建产物"""
        print("🧹 清理之前的构建...")
        if self.target_dir.exists():
            shutil.rmtree(self.target_dir)
        print("✅ 清理完成")
    
    def build_rust_engine(self, profile="release", features=None):
        """构建 Rust 引擎"""
        print(f"🔨 构建 Rust 引擎 (profile: {profile})...")
        
        # 构建命令
        cmd = ["maturin", "build"]
        
        if profile == "release":
            cmd.append("--release")
        elif profile == "edge-release":
            cmd.extend(["--profile", "edge-release"])
            
        if features:
            cmd.extend(["--features", ",".join(features)])
            
        # 设置环境变量
        env = os.environ.copy()
        
        # 如果是 ARM 架构（树莓派等），添加特殊配置
        if platform.machine().lower() in ['aarch64', 'armv7l']:
            print("🔧 检测到 ARM 架构，应用优化配置...")
            env["CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER"] = "aarch64-linux-gnu-gcc"
            
        try:
            subprocess.run(cmd, cwd=self.rust_dir, env=env, check=True)
            print("✅ Rust 引擎构建成功")
        except subprocess.CalledProcessError as e:
            print(f"❌ 构建失败: {e}")
            sys.exit(1)
    
    def install_wheel(self):
        """安装生成的 wheel 包"""
        print("📦 安装 Python 包...")
        
        # 查找最新的 wheel 文件
        wheels_dir = self.target_dir / "wheels"
        if not wheels_dir.exists():
            print("❌ 未找到 wheel 文件")
            sys.exit(1)
            
        wheel_files = list(wheels_dir.glob("*.whl"))
        if not wheel_files:
            print("❌ 未找到 wheel 文件")
            sys.exit(1)
            
        latest_wheel = max(wheel_files, key=os.path.getmtime)
        print(f"📦 安装: {latest_wheel.name}")
        
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            str(latest_wheel), "--force-reinstall"
        ], check=True)
        
        print("✅ 安装完成")
    
    def run_tests(self):
        """运行测试"""
        print("🧪 运行测试...")
        
        try:
            # Rust 测试
            subprocess.run(["cargo", "test"], cwd=self.rust_dir, check=True)
            
            # Python 集成测试
            test_script = """
import sys
try:
    from edge_gateway.rust_engine import EdgeFitEngine
    engine = EdgeFitEngine("./models/dummy_model.pt")
    print("✅ Python 绑定测试通过")
except Exception as e:
    print(f"❌ Python 绑定测试失败: {e}")
    sys.exit(1)
"""
            subprocess.run([sys.executable, "-c", test_script], check=True)
            
        except subprocess.CalledProcessError:
            print("❌ 测试失败")
            sys.exit(1)
    
    def benchmark(self):
        """运行性能基准测试"""
        print("⚡ 运行性能基准测试...")
        
        subprocess.run([
            "cargo", "bench", 
            "--", "--output-format", "pretty"
        ], cwd=self.rust_dir)
    
    def setup_development(self):
        """设置开发环境"""
        print("🛠️  设置开发环境...")
        
        # 安装开发依赖
        dev_deps = [
            "maturin",
            "pytest",
            "black",
            "mypy",
        ]
        
        subprocess.run([
            sys.executable, "-m", "pip", "install"
        ] + dev_deps, check=True)
        
        # 设置 git hooks（如果是 git 仓库）
        git_dir = self.project_root / ".git"
        if git_dir.exists():
            hooks_dir = git_dir / "hooks"
            pre_commit_hook = hooks_dir / "pre-commit"
            
            if not pre_commit_hook.exists():
                hooks_dir.mkdir(exist_ok=True)
                with open(pre_commit_hook, 'w') as f:
                    f.write("""#!/bin/sh
# 构建前检查
cd edge_gateway/rust_engine
cargo clippy -- -D warnings
cargo fmt -- --check
""")
                pre_commit_hook.chmod(0o755)
                print("✅ 设置 git pre-commit hook")
        
        print("✅ 开发环境设置完成")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="EdgeFit Rust Engine 构建脚本")
    parser.add_argument("--clean", action="store_true", help="清理构建")
    parser.add_argument("--profile", default="release", 
                       choices=["dev", "release", "edge-release"],
                       help="构建配置")
    parser.add_argument("--features", nargs="*", 
                       help="启用的特性")
    parser.add_argument("--test", action="store_true", help="运行测试")
    parser.add_argument("--benchmark", action="store_true", help="运行基准测试")
    parser.add_argument("--dev-setup", action="store_true", help="设置开发环境")
    parser.add_argument("--no-install", action="store_true", help="不安装构建结果")
    
    args = parser.parse_args()
    
    builder = RustEngineBuilder()
    
    if args.dev_setup:
        builder.setup_development()
        return
    
    # 检查依赖
    builder.check_dependencies()
    
    # 清理（如果需要）
    if args.clean:
        builder.clean_build()
    
    # 构建
    builder.build_rust_engine(
        profile=args.profile,
        features=args.features
    )
    
    # 安装
    if not args.no_install:
        builder.install_wheel()
    
    # 测试
    if args.test:
        builder.run_tests()
    
    # 基准测试
    if args.benchmark:
        builder.benchmark()
    
    print("🎉 构建完成！")

if __name__ == "__main__":
    main()