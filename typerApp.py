#!/usr/bin/env python3
import typer

def main(name: str):
    print(f"hola {name}")

if __name__ == "__main__":
    typer.run(main)