import java.awt.BorderLayout;

import javax.swing.JFrame;

public class FiguraVentana extends JFrame {

	
	public FiguraVentana(){
		super("Dibujar Figuras");
		this.setDefaultCloseOperation(EXIT_ON_CLOSE);
		this.setLocation(280, 50);
		FiguraPanel a=new FiguraPanel();
		FiguraControles b=new FiguraControles(a);
		this.add(a);
		this.add(b,BorderLayout.WEST);
		
		this.pack();
		
		this.setVisible(true);
		
	}
	
	public static void main(String[] args) {
		FiguraVentana a=new FiguraVentana();
		

	}

}
