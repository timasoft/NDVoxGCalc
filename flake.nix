{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    crane.url = "github:ipetkov/crane";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs = { self, nixpkgs, crane, flake-utils, rust-overlay, ... }:
    flake-utils.lib.eachDefaultSystem (system:
    let
      pkgs = import nixpkgs {
        inherit system;
        overlays = [ rust-overlay.overlays.default ];
      };

      rustToolchain = pkgs.rust-bin.nightly.latest.default.override {
        extensions = [ "rust-src" "rust-analyzer" "clippy" "rustfmt" ];
        targets = [ "wasm32-unknown-unknown" ];
      };

      craneLib = (crane.mkLib pkgs).overrideToolchain (p: rustToolchain);

      runtimeDeps = with pkgs; [
        libX11
        libXcursor
        libXrandr
        libXi
        libXext
        wayland
        libxkbcommon
        libGL
        vulkan-loader
        fontconfig
        freetype
        alsa-lib
        udev
      ];

      libPath = pkgs.lib.makeLibraryPath runtimeDeps;

      pkgConfigPath = pkgs.lib.makeSearchPathOutput "dev" "lib/pkgconfig" runtimeDeps;

      src = pkgs.lib.fileset.toSource {
        root = ./.;
        fileset = pkgs.lib.fileset.unions [
          ./Cargo.toml ./Cargo.lock ./Trunk.toml ./index.html ./src ./expr
          (pkgs.lib.fileset.maybeMissing ./.cargo)
          (pkgs.lib.fileset.maybeMissing ./assets)
        ];
      };

      cargoVendorDir = craneLib.vendorMultipleCargoDeps {
        inherit (craneLib.findCargoFiles src) cargoConfigs;
        cargoLockList = [
          ./Cargo.lock
          "${rustToolchain.passthru.availableComponents.rust-src}/lib/rustlib/src/rust/library/Cargo.lock"
        ];
      };

      nativeCargoArtifacts = craneLib.buildDepsOnly {
        inherit src;
        inherit cargoVendorDir;
        nativeBuildInputs = with pkgs; [ pkg-config ];
        buildInputs = runtimeDeps;

        doCheck = false;

        env = {
          PKG_CONFIG_PATH = pkgConfigPath;
          CARGO_PROFILE_RELEASE_LTO = "true";
          CARGO_PROFILE_RELEASE_CODEGEN_UNITS = "1";
        };
      };

      buildNative = fma: craneLib.buildPackage {
        pname = "hypervox";
        version = "0.1.0";
        inherit src;
        inherit cargoVendorDir;
        cargoArtifacts = nativeCargoArtifacts;

        buildInputs = runtimeDeps;
        nativeBuildInputs = with pkgs; [ pkg-config makeWrapper ];

        doCheck = false;

        cargoExtraArgs = pkgs.lib.optionalString fma "--features fma";

        env = {
          PKG_CONFIG_PATH = pkgConfigPath;
          CARGO_PROFILE_RELEASE_LTO = "true";
          CARGO_PROFILE_RELEASE_CODEGEN_UNITS = "1";
          CARGO_PROFILE_RELEASE_STRIP = "symbols";
        } // (pkgs.lib.optionalAttrs fma { RUSTFLAGS = "-C target-feature=+fma"; });

        postFixup = ''
          if [ -x "$out/bin/hypervox" ]; then
            wrapProgram "$out/bin/hypervox" \
              --prefix LD_LIBRARY_PATH : "${libPath}" \
              --prefix LD_LIBRARY_PATH : /run/opengl-driver/lib
          fi
        '';
      };

      native = buildNative false;

      native-fma = buildNative true;

      webCargoArtifacts = craneLib.buildDepsOnly {
        inherit src;
        inherit cargoVendorDir;
        CARGO_BUILD_TARGET = "wasm32-unknown-unknown";
        doCheck = false;
        CARGO_PROFILE_RELEASE_LTO = "true";
        CARGO_PROFILE_RELEASE_CODEGEN_UNITS = "1";
      };

      web = craneLib.buildTrunkPackage {
        pname = "hypervox-web";
        version = "0.1.0";
        inherit src;
        inherit cargoVendorDir;
        cargoArtifacts = webCargoArtifacts;

        CARGO_BUILD_TARGET = "wasm32-unknown-unknown";
        doCheck = false;

        wasm-bindgen-cli = pkgs.wasm-bindgen-cli_0_2_117;

        nativeBuildInputs = with pkgs; [ trunk binaryen lld pkg-config ];

        CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_LINKER = "${pkgs.lld}/bin/lld";

        preBuild = ''
          export WASM_BINDGEN_CACHE_DIR="$TMPDIR/wasm-bindgen"
          export XDG_CACHE_HOME="$TMPDIR/.cache"
          mkdir -p "$WASM_BINDGEN_CACHE_DIR" "$XDG_CACHE_HOME"
        '';

        postInstall = ''
          cat > "$out/_headers" <<'EOF'
/*
  Cross-Origin-Opener-Policy: same-origin
  Cross-Origin-Embedder-Policy: require-corp
EOF
        '';

        CARGO_PROFILE_RELEASE_LTO = "true";
        CARGO_PROFILE_RELEASE_CODEGEN_UNITS = "1";
      };
    in
    {
      packages = {
        inherit native native-fma web;
        default = native;
      };

      apps = {
        native = flake-utils.lib.mkApp {
          drv = native;
          exePath = "/bin/hypervox";
        };
        default = self.apps.${system}.native;
      };

      devShells.default = craneLib.devShell {
        inputsFrom = [ native web ];
        packages = with pkgs; [
          cargo rustc rustfmt clippy rust-analyzer
          trunk wasm-bindgen-cli binaryen vulkan-tools lld
          fish
        ];

        env = {
          LD_LIBRARY_PATH = "${libPath}:/run/opengl-driver/lib";
          RUST_SRC_PATH = "${rustToolchain}/lib/rustlib/src/rust/library";
          CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_LINKER = "${pkgs.lld}/bin/lld";
        };

        shellHook = ''
          if [ -z "$FISH_VERSION" ] && [ -z "$NO_AUTO_FISH" ] && [ -t 0 ]; then
            exec ${pkgs.fish}/bin/fish
          fi
        '';
      };
    });
}
