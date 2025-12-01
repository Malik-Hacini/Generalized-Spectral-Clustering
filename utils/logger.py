class ExperimentLogger:
    
    def __init__(self, verbose=True):
        self.verbose = verbose
    
    def info(self, message, indent=0):
        """Log informational messages (only shown when verbose=True)."""
        if self.verbose:
            print("  " * indent + message)
    
    def minimal(self, message):
        """Log minimal messages (always shown regardless of verbose setting)."""
        print(message)
    
    def error(self, message, indent=0):
        """Log error messages with a clear error symbol (always shown)."""
        print("  " * indent + "🔴 " + message)
    
    def success(self, message, indent=0):
        """Log success messages (always shown)."""
        print("  " * indent + "✅ " + message)
    
    def list_items(self, title, items, indent=0):
        if self.verbose:
            print("  " * indent + f"{title}:")
            for item in items:
                print("  " * (indent + 1) + f"• {item}")
    
    def section(self, title, indent=0):
        if self.verbose:
            print("  " * indent + f"\n{title}")
            print("  " * indent + "─" * len(title))

# Global logger instance
_logger = None

def get_logger():
    """Get the current experiment logger instance."""
    global _logger
    if _logger is None:
        _logger = ExperimentLogger()
    return _logger

def set_logger_verbose(verbose):
    """Set the global logger verbosity."""
    global _logger
    _logger = ExperimentLogger(verbose)
