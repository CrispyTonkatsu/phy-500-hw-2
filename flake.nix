{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs = { nixpkgs, ... }: {
    devShells = builtins.mapAttrs (system: pkgs: {
      default = pkgs.mkShell {
        packages = with pkgs; [
          (python3.withPackages (
            ps: with ps; [
              jupytext
              jupyter-collaboration

              sympy
              numpy
              matplotlib

              playwright
              nbconvert
            ]
          ))

          playwright
          playwright-driver

          jupyter

          basedpyright
          ruff
        ];
      };
    }) nixpkgs.legacyPackages;
  };
}
