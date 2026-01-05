{
  description = "Development environment for the E-Nose";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = {
    self,
    nixpkgs,
  }: let
    system = "x86_64-linux";
    pkgs = import nixpkgs {inherit system;};
    python = pkgs.python311;
  in {
    devShells.${system}.default = pkgs.mkShell {
      packages = with pkgs; [
        uv
        python
        docker-compose
        black
        isort
        ruff
      ];

      shellHook = ''
        export UV_PROJECT_ROOT="$PWD"
        export UV_PYTHON="${python}/bin/python3"
        export UV_NO_SYNC_PROGRESS=1
        if [ ! -d .venv ]; then
          echo "[flake] Creating virtualenv via uv..."
          uv sync --python "$UV_PYTHON"
        fi
        if [ -f .venv/bin/activate ]; then
          # shellcheck disable=SC1091
          source .venv/bin/activate
        fi
      '';
    };
  };
}
