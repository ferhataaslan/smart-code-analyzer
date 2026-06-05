import sys
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text
from src.database import database
from src.core import uploader

console = Console()

def create_layout():
    layout = Layout()
    layout.split_row(
        Layout(name="left"),
        Layout(name="right")
    )
    return layout

def main():
    console.clear()
    console.print(Panel("[bold cyan]Antigravity Smart Code Analyzer - Distributed Review Station[/bold cyan]"))
    
    source = Prompt.ask("Hangi kaynaktan devam edeceksin?", choices=["github", "hf", "owasp"], default="hf")
    
    while True:
        record = database.get_pending_record(source)
        
        if not record:
            console.print("[yellow]Bu kaynakta incelenecek 'pending' durumunda veri kalmadı![/yellow]")
            break
            
        layout = create_layout()
        
        # Raw Code Panel
        raw_text = Text(record['raw_code'])
        layout["left"].update(Panel(raw_text, title=f"Raw Code (ID: {record['id']} - {record['source']})", border_style="red"))
        
        # Processed Code Panel
        proc_text = Text(record['processed_code'])
        layout["right"].update(Panel(proc_text, title="Tree-Sitter Processed Code", border_style="green"))
        
        console.clear()
        console.print(layout)
        
        console.print("\n[bold]Hotkeys:[/bold] [green]A[/green]: Onayla | [red]R[/red]: Reddet | [yellow]Q[/yellow]: Çıkış ve Kaydet")
        
        choice = Prompt.ask("Seçiminiz", choices=["A", "a", "R", "r", "Q", "q"])
        choice = choice.upper()
        
        if choice == 'A':
            database.update_status(record['id'], 'approved')
            console.print("[green]Kayıt onaylandı.[/green]")
        elif choice == 'R':
            database.update_status(record['id'], 'rejected')
            console.print("[red]Kayıt reddedildi.[/red]")
        elif choice == 'Q':
            console.print("[yellow]Çıkış yapılıyor...[/yellow]")
            break

    # ── Push Kontrolü — döngüden HER çıkışta (Q veya pending bitti) çalışır ──
    approved_count = uploader.count_approved_records()
    if approved_count < 1:
        console.print("[dim]Onaylı kayıt yok. Yükleme atlandı.[/dim]")
    else:
        push_choice = Prompt.ask(
            f"[bold yellow]Toplam {approved_count} onaylı kayıt mevcut. "
            f"Hugging Face Hub'a push edilsin mi?[/bold yellow]",
            choices=["Y", "y", "N", "n"], default="N"
        )
        if push_choice.upper() == 'Y':
            console.print("[yellow]MLOps uploader tetikleniyor...[/yellow]")
            uploader.export_and_push()
        else:
            console.print("[dim]Yükleme atlandı.[/dim]")

if __name__ == "__main__":
    main()
