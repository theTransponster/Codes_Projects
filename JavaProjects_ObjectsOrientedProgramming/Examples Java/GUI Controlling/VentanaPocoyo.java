import java.awt.BorderLayout;

import javax.swing.JFrame;

public class VentanaPocoyo extends JFrame {

	public VentanaPocoyo(){
		super("Vamoosh De Pocoyo");
		PocoyoPanel p=new PocoyoPanel();
		this.add(p);
		this.add(new PocoyoControles(p),BorderLayout.WEST);
		this.pack();
		this.setDefaultCloseOperation(EXIT_ON_CLOSE);
		this.setVisible(true);
	}
	public static void main(String[] args) {
		VentanaPocoyo nuevaPocoyo=new VentanaPocoyo();
	}

}
