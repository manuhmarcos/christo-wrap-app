import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:share_plus/share_plus.dart';
import 'package:gallery_saver/gallery_saver.dart';
import 'dart:io';
import 'package:path_provider/path_provider.dart';

class ResultScreen extends StatefulWidget {
  final Uint8List originalBytes;
  final Uint8List resultBytes;

  const ResultScreen({
    super.key,
    required this.originalBytes,
    required this.resultBytes,
  });

  @override
  State<ResultScreen> createState() => _ResultScreenState();
}

class _ResultScreenState extends State<ResultScreen> {
  bool _showOriginal = false;
  bool _isSaving = false;

  Future<void> _saveToGallery() async {
    setState(() => _isSaving = true);
    try {
      final dir = await getTemporaryDirectory();
      final file = File('${dir.path}/christo_wrap_${DateTime.now().millisecondsSinceEpoch}.jpg');
      await file.writeAsBytes(widget.resultBytes);
      final success = await GallerySaver.saveImage(file.path, albumName: 'Christo Wrap');
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(success == true ? 'Guardado en galeria' : 'Error al guardar'),
          backgroundColor: success == true ? Colors.green[800] : Colors.red[800],
        ),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Error: $e'),
          backgroundColor: Colors.red[800],
        ),
      );
    } finally {
      if (mounted) setState(() => _isSaving = false);
    }
  }

  Future<void> _share() async {
    try {
      final dir = await getTemporaryDirectory();
      final file = File('${dir.path}/christo_wrap_share.jpg');
      await file.writeAsBytes(widget.resultBytes);
      await Share.shareXFiles(
        [XFile(file.path)],
        text: 'Wrapped al estilo Christo y Jeanne-Claude 🎨',
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error al compartir: $e')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF1A1209),
      appBar: AppBar(
        backgroundColor: const Color(0xFF1A1209),
        foregroundColor: const Color(0xFFD4A843),
        title: const Text(
          'RESULTADO',
          style: TextStyle(letterSpacing: 3, fontSize: 16),
        ),
        centerTitle: true,
      ),
      body: SafeArea(
        child: Column(
          children: [
            // Image viewer
            Expanded(
              child: GestureDetector(
                onTapDown: (_) => setState(() => _showOriginal = true),
                onTapUp: (_) => setState(() => _showOriginal = false),
                onTapCancel: () => setState(() => _showOriginal = false),
                child: Stack(
                  children: [
                    // Result image
                    Positioned.fill(
                      child: Image.memory(
                        _showOriginal ? widget.originalBytes : widget.resultBytes,
                        fit: BoxFit.contain,
                      ),
                    ),
                    // Hint label
                    Positioned(
                      bottom: 16,
                      left: 0,
                      right: 0,
                      child: Center(
                        child: AnimatedOpacity(
                          opacity: _showOriginal ? 0 : 0.8,
                          duration: const Duration(milliseconds: 300),
                          child: Container(
                            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                            decoration: BoxDecoration(
                              color: Colors.black54,
                              borderRadius: BorderRadius.circular(20),
                            ),
                            child: const Text(
                              'Manten presionado para ver original',
                              style: TextStyle(color: Colors.white70, fontSize: 12),
                            ),
                          ),
                        ),
                      ),
                    ),
                    if (_showOriginal)
                      Positioned(
                        top: 16,
                        left: 0,
                        right: 0,
                        child: Center(
                          child: Container(
                            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                            decoration: BoxDecoration(
                              color: Colors.black54,
                              borderRadius: BorderRadius.circular(20),
                            ),
                            child: const Text(
                              'ORIGINAL',
                              style: TextStyle(
                                color: Colors.white,
                                fontSize: 12,
                                letterSpacing: 2,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                          ),
                        ),
                      ),
                  ],
                ),
              ),
            ),
            // Action buttons
            Padding(
              padding: const EdgeInsets.all(24),
              child: Row(
                children: [
                  Expanded(
                    child: _ResultButton(
                      icon: _isSaving ? null : Icons.download,
                      label: _isSaving ? 'GUARDANDO...' : 'GUARDAR',
                      onTap: _isSaving ? null : _saveToGallery,
                      primary: false,
                    ),
                  ),
                  const SizedBox(width: 16),
                  Expanded(
                    child: _ResultButton(
                      icon: Icons.share,
                      label: 'COMPARTIR',
                      onTap: _share,
                      primary: true,
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _ResultButton extends StatelessWidget {
  final IconData? icon;
  final String label;
  final VoidCallback? onTap;
  final bool primary;

  const _ResultButton({
    required this.icon,
    required this.label,
    required this.onTap,
    required this.primary,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 16),
        decoration: BoxDecoration(
          color: primary ? const Color(0xFFD4A843) : Colors.transparent,
          border: primary
              ? null
              : Border.all(color: const Color(0xFF4A3A1A), width: 1.5),
          borderRadius: BorderRadius.circular(14),
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            if (icon != null) ...[
              Icon(
                icon,
                color: primary ? const Color(0xFF1A1209) : const Color(0xFF9E8B6A),
                size: 20,
              ),
              const SizedBox(width: 8),
            ],
            Text(
              label,
              style: TextStyle(
                color: primary ? const Color(0xFF1A1209) : const Color(0xFF9E8B6A),
                fontWeight: FontWeight.bold,
                fontSize: 13,
                letterSpacing: 1.5,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
